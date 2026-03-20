from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..engine.config import GameConfig
from ..engine.game import GameState
from ..rendering.numpy_renderer import NumpyRenderer
from .rewards import RewardConfig, compute_reward


class SlitherEnv(gym.Env):
    """
    A slither.io Gymnasium environment suitable for DreamerV3 training.

    Observation: (obs_size, obs_size, 3) uint8 RGB image, ego-centric view.
    Action: Discrete(6) — 0=straight, 1=left, 2=right, 3=boost straight, 4=boost left, 5=boost right.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        num_npcs: int = 4,
        arena_radius: float = 500.0,
        max_steps: int = 4000,
        obs_size: int = 64,
        viewport_radius: float = 120.0,
        reward_config: RewardConfig | None = None,
    ):
        super().__init__()

        self.render_mode = render_mode

        self.game_config = GameConfig(
            arena_radius=arena_radius,
            num_npcs=num_npcs,
            max_steps=max_steps,
            obs_size=obs_size,
            viewport_radius=viewport_radius,
        )

        self.reward_config = reward_config or RewardConfig()

        # Spaces
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(obs_size, obs_size, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(6)

        # Will be initialized on reset
        self._state: GameState | None = None
        self._renderer: NumpyRenderer | None = None
        self._pygame_renderer = None
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Use gymnasium's seeded RNG to derive a deterministic seed for the engine
        engine_seed = int(self.np_random.integers(0, 2**31))
        self._state = GameState(self.game_config, seed=engine_seed)
        self._renderer = NumpyRenderer(self.game_config)
        self._step_count = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state is not None, "Call reset() before step()"

        events = self._state.step(int(action))
        self._step_count += 1

        reward = compute_reward(events, self.reward_config)
        terminated = events.get("died", False)
        truncated = self._step_count >= self.game_config.max_steps

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        assert self._state is not None and self._renderer is not None
        return self._renderer.render(self._state)

    def _get_info(self) -> dict:
        assert self._state is not None
        player = self._state.player
        return {
            "length": player.length,
            "score": player.score,
            "alive": player.alive,
            "step": self._step_count,
            "alive_npcs": sum(1 for s in self._state.snakes[1:] if s.alive),
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._get_obs()
        elif self.render_mode == "human":
            return self._render_human()
        return None

    def _render_human(self) -> None:
        """Render to a pygame window for human viewing."""
        try:
            import pygame
        except ImportError:
            raise ImportError(
                "pygame is required for human rendering. "
                "Install it with: pip install pygame"
            )

        if self._pygame_renderer is None:
            pygame.init()
            self._pygame_screen = pygame.display.set_mode((512, 512))
            pygame.display.set_caption("Slither Gym")
            self._pygame_clock = pygame.time.Clock()
            self._pygame_renderer = True

        obs = self._get_obs()
        # Scale up to 512x512
        surface = pygame.surfarray.make_surface(
            np.transpose(
                np.repeat(np.repeat(obs, 8, axis=0), 8, axis=1),
                (1, 0, 2),
            )
        )
        self._pygame_screen.blit(surface, (0, 0))
        pygame.display.flip()
        self._pygame_clock.tick(self.metadata["render_fps"])

        # Process pygame events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        if self._pygame_renderer is not None:
            import pygame
            pygame.quit()
            self._pygame_renderer = None
