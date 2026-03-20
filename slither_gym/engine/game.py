from __future__ import annotations

import numpy as np

from .config import GameConfig
from .food import FoodManager
from .snake import Snake


class GameState:
    """Core game simulation: snakes, food, collisions, NPC AI."""

    def __init__(self, config: GameConfig, seed: int | None = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

        # Player snake (index 0) + NPCs
        self.snakes: list[Snake] = []
        self._init_snakes()

        # Food
        self.food = FoodManager(config, self.rng)

        # Per-step event tracking (for rewards)
        self.events: dict = {}

    def _init_snakes(self):
        total = 1 + self.config.num_npcs
        self.snakes.clear()
        for i in range(total):
            snake = self._spawn_snake(i)
            self.snakes.append(snake)

    def _spawn_snake(self, snake_id: int) -> Snake:
        r = self.config.arena_radius * 0.7
        angle = self.rng.random() * 2 * np.pi
        radius = np.sqrt(self.rng.random()) * r
        pos = np.array([np.cos(angle) * radius, np.sin(angle) * radius], dtype=np.float32)
        direction = self.rng.random() * 2 * np.pi
        return Snake(
            position=pos,
            direction=direction,
            length=self.config.initial_length,
            max_length=self.config.max_length,
            segment_spacing=self.config.segment_spacing,
            snake_id=snake_id,
        )

    @property
    def player(self) -> Snake:
        return self.snakes[0]

    def step(self, player_action: int) -> dict:
        """Advance one simulation step. Returns event dict."""
        self.step_count += 1
        self.events = {
            "food_eaten": 0.0,
            "killed_opponent": 0,
            "died": False,
            "boosting": False,
            "length": self.player.length,
        }

        # Apply player action
        self._apply_player_action(player_action)

        # NPC AI
        self._step_npcs()

        # Move all alive snakes
        for snake in self.snakes:
            if not snake.alive:
                continue
            if snake.boosting and snake.length > self.config.initial_length:
                # Boost: take 2 sub-steps at base speed to maintain segment spacing
                # but move twice as far overall
                for _ in range(2):
                    snake.move(self.config.base_speed)
                # Accumulate fractional boost cost; drop pellet when a full segment is lost
                snake._boost_debt += self.config.boost_mass_cost
                if snake._boost_debt >= 1.0:
                    tail_pos = snake.positions[snake.tail_idx].copy()
                    self.food.spawn_boost_pellet(tail_pos)
                    snake.shrink(1)
                    snake._boost_debt -= 1.0
            else:
                snake.move(self.config.base_speed)

        # Food eating
        for snake in self.snakes:
            if not snake.alive:
                continue
            eaten = self.food.check_eat(snake.head_pos, self.config.head_radius)
            if eaten > 0:
                snake.grow(eaten)
                snake.score += eaten
                if snake.id == 0:
                    self.events["food_eaten"] = eaten

        # Collision detection
        self._check_collisions()

        # Food respawn
        self.food.step()

        # NPC respawn
        self._respawn_dead_npcs()

        return self.events

    def _apply_player_action(self, action: int):
        """Discrete(6): 0=straight, 1=left, 2=right, 3=boost straight, 4=boost left, 5=boost right."""
        player = self.player
        if not player.alive:
            return

        turn_action = action % 3
        wants_boost = action >= 3

        if turn_action == 1:
            player.turn(-1.0, self.config.turn_rate)
        elif turn_action == 2:
            player.turn(1.0, self.config.turn_rate)

        # Boost only if snake is long enough
        min_boost_length = self.config.initial_length
        player.boosting = wants_boost and player.length > min_boost_length
        self.events["boosting"] = player.boosting

    def _step_npcs(self):
        """Simple NPC behavior: wander + eat, gently steer toward nearby food."""
        for snake in self.snakes[1:]:
            if not snake.alive:
                continue

            snake.boosting = False

            # Find nearest food and steer toward it
            active_food = self.food.positions[self.food.active]
            if len(active_food) > 0:
                diffs = active_food - snake.head_pos
                dists_sq = np.sum(diffs * diffs, axis=1)
                nearest_idx = np.argmin(dists_sq)
                target = active_food[nearest_idx]
                nearest_dist_sq = dists_sq[nearest_idx]

                # Desired angle to target
                desired = np.arctan2(
                    target[1] - snake.head_pos[1],
                    target[0] - snake.head_pos[0],
                )
                # Angle difference
                diff = (desired - snake.direction + np.pi) % (2 * np.pi) - np.pi
                turn_amount = np.clip(diff / self.config.turn_rate, -1.0, 1.0)
                snake.turn(turn_amount, self.config.turn_rate)

                # Occasionally boost toward nearby food clusters or when chasing
                if (snake.length > self.config.initial_length + 5
                        and nearest_dist_sq < 60.0 ** 2
                        and self.rng.random() < 0.3):
                    snake.boosting = True
            else:
                # Random wander
                snake.turn(self.rng.uniform(-0.3, 0.3), self.config.turn_rate)

            # Occasionally add randomness to make NPCs less predictable
            if self.rng.random() < 0.15:
                snake.turn(self.rng.uniform(-0.5, 0.5), self.config.turn_rate)

    def _check_collisions(self):
        """Check head-to-body and boundary collisions."""
        arena_r = self.config.arena_radius

        for i, snake in enumerate(self.snakes):
            if not snake.alive:
                continue

            # Boundary check (circular arena)
            dist_from_center = np.sqrt(np.sum(snake.head_pos ** 2))
            if dist_from_center > arena_r:
                self._kill_snake(i, killer_id=-1)
                continue

            # Head-to-body collision with other snakes
            for j, other in enumerate(self.snakes):
                if i == j or not other.alive:
                    continue
                if self._head_hits_body(snake, other):
                    self._kill_snake(i, killer_id=j)
                    break

    def _head_hits_body(self, snake: Snake, other: Snake) -> bool:
        """Check if snake's head collides with other's body segments (not head)."""
        head = snake.head_pos
        segments = other.active_segments()

        # Skip first 3 segments (head area) to avoid self-like false positives
        if len(segments) <= 3:
            return False
        body = segments[3:]

        diffs = body - head
        dists_sq = np.sum(diffs * diffs, axis=1)
        collision_dist = self.config.head_radius + self.config.body_radius
        return bool(np.any(dists_sq < collision_dist ** 2))

    def _kill_snake(self, idx: int, killer_id: int):
        snake = self.snakes[idx]
        if not snake.alive:
            return
        snake.kill()

        # Drop food at death location
        segments = snake.active_segments()
        self.food.spawn_death_food(segments, self.config.death_food_fraction)

        if idx == 0:
            self.events["died"] = True
        if killer_id == 0:
            self.events["killed_opponent"] += 1

        # Set respawn timer for NPCs
        if idx > 0:
            snake.respawn_timer = self.config.npc_respawn_delay

    def _respawn_dead_npcs(self):
        for i, snake in enumerate(self.snakes[1:], start=1):
            if snake.alive:
                continue
            snake.respawn_timer -= 1
            if snake.respawn_timer <= 0:
                self.snakes[i] = self._spawn_snake(i)
