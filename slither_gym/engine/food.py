from __future__ import annotations

import numpy as np

from .config import GameConfig


class FoodManager:
    """Manages food pellets with pre-allocated masked arrays."""

    def __init__(self, config: GameConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

        self.positions = np.zeros((config.max_food, 2), dtype=np.float32)
        self.values = np.full(config.max_food, config.food_value, dtype=np.float32)
        self.active = np.zeros(config.max_food, dtype=bool)
        self.colors = np.zeros((config.max_food, 3), dtype=np.uint8)

        # Spawn initial food
        self._spawn_n(config.initial_food)

    def _random_pos_in_arena(self, n: int) -> np.ndarray:
        """Generate n random positions within the circular arena."""
        r = self.config.arena_radius
        # Uniform in disk: sqrt for uniform area distribution
        radii = np.sqrt(self.rng.random(n, dtype=np.float32)) * r * 0.95
        angles = self.rng.random(n, dtype=np.float32) * 2 * np.pi
        pos = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
        return pos

    def _spawn_n(self, n: int):
        """Spawn up to n food pellets in inactive slots."""
        inactive = np.where(~self.active)[0]
        count = min(n, len(inactive))
        if count == 0:
            return
        slots = inactive[:count]
        self.positions[slots] = self._random_pos_in_arena(count)
        self.values[slots] = self.config.food_value
        self.active[slots] = True
        # Random bright colors
        self.colors[slots] = self.rng.integers(128, 256, size=(count, 3), dtype=np.uint8)

    def spawn_death_food(self, positions: np.ndarray, fraction: float):
        """Scatter food along the snake's body where it died, like slither.io."""
        n = max(1, int(len(positions) * fraction))
        n = min(n, len(positions))
        inactive = np.where(~self.active)[0]
        count = min(n, len(inactive))
        if count == 0:
            return
        # Pick evenly spaced segment positions along the body
        indices = np.linspace(0, len(positions) - 1, count, dtype=int)
        slots = inactive[:count]
        self.positions[slots] = positions[indices]
        # Small jitter to spread pellets slightly around the body line
        self.positions[slots] += self.rng.normal(0, 1.5, size=(count, 2)).astype(np.float32)
        self.values[slots] = self.config.food_value * 2  # death food is richer
        self.active[slots] = True
        # Use a consistent bright color per snake death (like slither.io)
        base_color = self.rng.integers(180, 256, size=(1, 3), dtype=np.uint8)
        # Slight per-pellet variation
        variation = self.rng.integers(-20, 20, size=(count, 3), dtype=np.int16)
        self.colors[slots] = np.clip(
            base_color.astype(np.int16) + variation, 128, 255
        ).astype(np.uint8)

    def spawn_boost_pellet(self, position: np.ndarray):
        """Spawn a single food pellet at the given position (from boosting)."""
        inactive = np.where(~self.active)[0]
        if len(inactive) == 0:
            return
        slot = inactive[0]
        self.positions[slot] = position
        self.values[slot] = self.config.food_value
        self.active[slot] = True
        self.colors[slot] = self.rng.integers(180, 256, size=3, dtype=np.uint8)

    def step(self):
        """Respawn food each step, but cap at initial_food to leave room for boost/death pellets."""
        current = int(np.sum(self.active))
        if current < self.config.initial_food:
            self._spawn_n(min(self.config.food_respawn_rate, self.config.initial_food - current))

    def check_eat(self, head_pos: np.ndarray, eat_radius: float) -> float:
        """Check if head_pos eats any food. Returns total value eaten."""
        if not np.any(self.active):
            return 0.0
        active_mask = self.active
        diffs = self.positions[active_mask] - head_pos
        dists_sq = np.sum(diffs * diffs, axis=1)
        threshold_sq = (eat_radius + self.config.food_radius) ** 2
        eaten_local = dists_sq < threshold_sq

        if not np.any(eaten_local):
            return 0.0

        # Map back to global indices
        active_indices = np.where(active_mask)[0]
        eaten_global = active_indices[eaten_local]

        total_value = float(np.sum(self.values[eaten_global]))
        self.active[eaten_global] = False
        return total_value
