from __future__ import annotations

import numpy as np


class Snake:
    """A snake with ring-buffer segment storage for O(1) movement."""

    def __init__(
        self,
        position: np.ndarray,
        direction: float,
        length: int,
        max_length: int,
        segment_spacing: float,
        snake_id: int = 0,
    ):
        self.id = snake_id
        self.max_length = max_length
        self.segment_spacing = segment_spacing

        # Ring buffer for segment positions
        self.positions = np.zeros((max_length, 2), dtype=np.float32)
        self.direction = float(direction)
        self.length = length
        self.alive = True
        self.boosting = False
        self.score = 0.0
        self.respawn_timer = 0
        self._boost_debt = 0.0  # fractional boost cost accumulator

        # Initialize segments trailing behind the head
        dx = -np.cos(direction) * segment_spacing
        dy = -np.sin(direction) * segment_spacing
        for i in range(length):
            self.positions[i, 0] = position[0] + dx * i
            self.positions[i, 1] = position[1] + dy * i

        # head is at index (length - 1) after init; we track it explicitly
        self.head_idx = length - 1
        self.tail_idx = 0

    @property
    def head_pos(self) -> np.ndarray:
        return self.positions[self.head_idx]

    def active_segments(self) -> np.ndarray:
        """Return (length, 2) array of active segment positions, head-first."""
        idx = self.head_idx
        indices = np.arange(self.length)
        ring_indices = (idx - indices) % self.max_length
        return self.positions[ring_indices]

    def move(self, speed: float):
        """Advance the head one step in the current direction."""
        if not self.alive:
            return
        new_head = self.head_pos.copy()
        new_head[0] += np.cos(self.direction) * speed
        new_head[1] += np.sin(self.direction) * speed

        # Advance head pointer
        self.head_idx = (self.head_idx + 1) % self.max_length
        self.positions[self.head_idx] = new_head

        # If not growing, advance tail
        if self.length >= self.max_length:
            self.tail_idx = (self.tail_idx + 1) % self.max_length
        else:
            # The ring buffer has room; but we only keep `length` active
            pass

        # Tail advances unless we just grew
        # (growth is handled externally by incrementing self.length)
        self.tail_idx = (self.head_idx - self.length + 1) % self.max_length

    def turn(self, amount: float, turn_rate: float):
        """Turn by `amount` in [-1, 1], scaled by turn_rate."""
        self.direction += amount * turn_rate
        # Normalize to [-pi, pi]
        self.direction = (self.direction + np.pi) % (2 * np.pi) - np.pi

    def grow(self, amount: float):
        """Grow by `amount` segments (can be fractional, rounded)."""
        add = int(round(amount))
        self.length = min(self.length + add, self.max_length)

    def shrink(self, amount: float):
        """Shrink by `amount` segments."""
        lose = int(round(amount))
        self.length = max(self.length - lose, 3)  # minimum viable snake
        self.tail_idx = (self.head_idx - self.length + 1) % self.max_length

    def kill(self):
        self.alive = False
