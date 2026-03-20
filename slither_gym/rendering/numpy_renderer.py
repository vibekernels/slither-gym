from __future__ import annotations

import numpy as np

from ..engine.config import GameConfig
from ..engine.game import GameState


# Slither.io-style dark blue color palette
BG_COLOR = np.array([20, 28, 45], dtype=np.uint8)
HEX_LINE_COLOR = np.array([35, 50, 75], dtype=np.uint8)
HEX_FILL_COLOR = np.array([22, 32, 50], dtype=np.uint8)
PLAYER_HEAD_COLOR = np.array([80, 255, 80], dtype=np.uint8)
PLAYER_BODY_COLOR = np.array([50, 220, 50], dtype=np.uint8)
PLAYER_GLOW_COLOR = np.array([30, 100, 30], dtype=np.uint8)
NPC_COLORS = [
    (np.array([255, 100, 100], dtype=np.uint8), np.array([200, 60, 60], dtype=np.uint8)),
    (np.array([100, 200, 255], dtype=np.uint8), np.array([60, 140, 200], dtype=np.uint8)),
    (np.array([255, 255, 100], dtype=np.uint8), np.array([200, 200, 60], dtype=np.uint8)),
    (np.array([255, 100, 255], dtype=np.uint8), np.array([200, 60, 200], dtype=np.uint8)),
    (np.array([100, 255, 255], dtype=np.uint8), np.array([60, 200, 200], dtype=np.uint8)),
    (np.array([255, 180, 100], dtype=np.uint8), np.array([200, 130, 60], dtype=np.uint8)),
    (np.array([180, 100, 255], dtype=np.uint8), np.array([130, 60, 200], dtype=np.uint8)),
    (np.array([255, 150, 150], dtype=np.uint8), np.array([200, 100, 100], dtype=np.uint8)),
]
BOUNDARY_COLOR = np.array([180, 50, 50], dtype=np.uint8)


class NumpyRenderer:
    """Renders ego-centric observations as NumPy RGB arrays in slither.io style."""

    def __init__(self, config: GameConfig):
        self.config = config
        self.size = config.obs_size
        self.vr = config.viewport_radius

        # Pre-compute pixel coordinate grid (centered on 0,0)
        half = self.size / 2
        y_coords, x_coords = np.mgrid[0:self.size, 0:self.size]
        # Map pixel coords to world-relative coords
        self.px = (x_coords.astype(np.float32) - half + 0.5) / half * self.vr
        self.py = (y_coords.astype(np.float32) - half + 0.5) / half * self.vr

    def render(self, state: GameState) -> np.ndarray:
        """Render ego-centric view around the player's head."""
        img = np.full((self.size, self.size, 3), BG_COLOR, dtype=np.uint8)

        player = state.player
        if not player.alive:
            return img

        cx, cy = player.head_pos

        # Draw hexagonal grid background
        self._draw_hex_grid(img, cx, cy)

        # Draw arena boundary
        self._draw_boundary(img, cx, cy)

        # Draw food (glowing pellets)
        self._draw_food(img, state, cx, cy)

        # Draw NPC snakes
        for i, snake in enumerate(state.snakes[1:]):
            if not snake.alive:
                continue
            head_color, body_color = NPC_COLORS[i % len(NPC_COLORS)]
            self._draw_snake(img, snake, cx, cy, body_color, head_color, snake.boosting)

        # Draw player snake (on top)
        self._draw_snake(img, player, cx, cy, PLAYER_BODY_COLOR, PLAYER_HEAD_COLOR, player.boosting)

        return img

    def _draw_hex_grid(self, img: np.ndarray, cx: float, cy: float):
        """Draw a hexagonal grid pattern like slither.io."""
        hex_size = 30.0  # radius of each hexagon in world units

        # World-space coordinates of each pixel
        wx = self.px + cx
        wy = self.py + cy

        # Hex grid math: convert world coords to axial hex coordinates
        # Using pointy-top hexagons
        sqrt3 = np.float32(1.7320508)
        q = (sqrt3 / 3.0 * wx - 1.0 / 3.0 * wy) / hex_size
        r = (2.0 / 3.0 * wy) / hex_size

        # Round to nearest hex center (cube rounding)
        s = -q - r
        qi = np.round(q).astype(np.int32)
        ri = np.round(r).astype(np.int32)
        si = np.round(s).astype(np.int32)

        q_diff = np.abs(qi.astype(np.float32) - q)
        r_diff = np.abs(ri.astype(np.float32) - r)
        s_diff = np.abs(si.astype(np.float32) - s)

        # Fix rounding: the component with largest diff gets reset
        fix_q = (q_diff > r_diff) & (q_diff > s_diff)
        fix_r = (~fix_q) & (r_diff > s_diff)

        qi[fix_q] = (-ri - si)[fix_q]
        ri[fix_r] = (-qi - si)[fix_r]

        # Convert hex center back to world coords
        hex_cx = hex_size * (sqrt3 * qi.astype(np.float32) + sqrt3 / 2.0 * ri.astype(np.float32))
        hex_cy = hex_size * (3.0 / 2.0 * ri.astype(np.float32))

        # Distance from pixel to its hex center
        dx = wx - hex_cx
        dy = wy - hex_cy

        # Distance to hex edge (approximate using max of 3 axes for pointy-top)
        # For a pointy-top hex, the distance to edge can be approximated
        abs_dx = np.abs(dx)
        abs_dy = np.abs(dy)
        # Hex edge distance using the hex norm
        dist_to_edge = np.maximum(
            abs_dy * (2.0 / 3.0),
            abs_dy * (1.0 / 3.0) + abs_dx * (sqrt3 / 3.0)
        )

        # Normalize by hex size
        edge_ratio = dist_to_edge / hex_size

        # Pixels near the edge of their hex cell = grid lines
        edge_width = 0.85  # threshold: closer to 1.0 = thinner lines
        edge_mask = edge_ratio > edge_width

        # Fill hexagons with slightly lighter color, edges with line color
        img[~edge_mask] = HEX_FILL_COLOR
        img[edge_mask] = HEX_LINE_COLOR

    def _draw_boundary(self, img: np.ndarray, cx: float, cy: float):
        """Draw arena boundary circle."""
        wx = self.px + cx
        wy = self.py + cy
        dist = np.sqrt(wx * wx + wy * wy)
        r = self.config.arena_radius
        boundary_mask = np.abs(dist - r) < 3.0
        img[boundary_mask] = BOUNDARY_COLOR
        # Shade outside arena
        outside_mask = dist > r
        img[outside_mask] = img[outside_mask] // 3

    def _draw_food(self, img: np.ndarray, state: GameState, cx: float, cy: float):
        """Draw glowing food pellets."""
        food = state.food
        active = food.active
        if not np.any(active):
            return

        positions = food.positions[active]
        colors = food.colors[active]

        # Convert food positions to pixel space
        rel_x = positions[:, 0] - cx
        rel_y = positions[:, 1] - cy

        # Cull food outside viewport (with margin)
        margin = self.vr + self.config.food_radius
        in_view = (np.abs(rel_x) < margin) & (np.abs(rel_y) < margin)
        if not np.any(in_view):
            return

        rel_x = rel_x[in_view]
        rel_y = rel_y[in_view]
        colors = colors[in_view]

        # Convert to pixel coordinates
        half = self.size / 2
        px = (rel_x / self.vr * half + half).astype(np.int32)
        py = (rel_y / self.vr * half + half).astype(np.int32)

        # Food radius in pixels (small glowing dots)
        fr = max(1, int(self.config.food_radius / self.vr * half))

        for k in range(len(px)):
            x, y = px[k], py[k]
            # Draw glow (larger, dimmer circle)
            gr = fr + 1
            x0, x1 = max(0, x - gr), min(self.size, x + gr + 1)
            y0, y1 = max(0, y - gr), min(self.size, y + gr + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.ogrid[y0:y1, x0:x1]
            glow_mask = (xx - x) ** 2 + (yy - y) ** 2 <= gr ** 2
            glow_color = colors[k].astype(np.int16) // 3
            region = img[y0:y1, x0:x1]
            blended = np.clip(
                region.astype(np.int16) + glow_color, 0, 255
            ).astype(np.uint8)
            region[glow_mask] = blended[glow_mask]

            # Draw core (bright center)
            x0, x1 = max(0, x - fr), min(self.size, x + fr + 1)
            y0, y1 = max(0, y - fr), min(self.size, y + fr + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.ogrid[y0:y1, x0:x1]
            core_mask = (xx - x) ** 2 + (yy - y) ** 2 <= fr ** 2
            img[y0:y1, x0:x1][core_mask] = colors[k]

    def _draw_snake(
        self,
        img: np.ndarray,
        snake,
        cx: float,
        cy: float,
        body_color: np.ndarray,
        head_color: np.ndarray,
        boosting: bool = False,
    ):
        """Draw a snake with slither.io-style segmented appearance."""
        segments = snake.active_segments()
        half = self.size / 2

        # Convert to pixel space
        rel_x = segments[:, 0] - cx
        rel_y = segments[:, 1] - cy

        # Cull segments outside viewport
        margin = self.vr + self.config.body_radius * 2
        in_view = (np.abs(rel_x) < margin) & (np.abs(rel_y) < margin)

        px_all = (rel_x / self.vr * half + half).astype(np.int32)
        py_all = (rel_y / self.vr * half + half).astype(np.int32)

        # Body radius in pixels
        br = max(1, int(self.config.body_radius / self.vr * half))
        hr = max(2, int(self.config.head_radius / self.vr * half))

        # When boosting, brighten colors for glow effect
        if boosting:
            body_color = np.clip(body_color.astype(np.int16) + 50, 0, 255).astype(np.uint8)
            head_color = np.clip(head_color.astype(np.int16) + 50, 0, 255).astype(np.uint8)

        # Compute a darker shade for segment ridges
        dark_body = np.clip(body_color.astype(np.int16) * 2 // 3, 0, 255).astype(np.uint8)

        # Draw glow aura when boosting (larger, dimmer circles behind each segment)
        if boosting:
            glow_color = body_color.astype(np.int16) // 4
            gr = br + 2
            for k in range(len(segments) - 1, -1, -1):
                if not in_view[k]:
                    continue
                x, y = px_all[k], py_all[k]
                x0, x1 = max(0, x - gr), min(self.size, x + gr + 1)
                y0, y1 = max(0, y - gr), min(self.size, y + gr + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                yy, xx = np.ogrid[y0:y1, x0:x1]
                glow_mask = (xx - x) ** 2 + (yy - y) ** 2 <= gr ** 2
                region = img[y0:y1, x0:x1]
                blended = np.clip(region.astype(np.int16) + glow_color, 0, 255).astype(np.uint8)
                region[glow_mask] = blended[glow_mask]

        # Draw body segments (tail-first so head draws on top)
        for k in range(len(segments) - 1, -1, -1):
            if not in_view[k]:
                continue
            x, y = px_all[k], py_all[k]
            r = hr if k == 0 else br
            is_head = k == 0

            x0, x1 = max(0, x - r - 1), min(self.size, x + r + 2)
            y0, y1 = max(0, y - r - 1), min(self.size, y + r + 2)
            if x0 >= x1 or y0 >= y1:
                continue

            yy, xx = np.ogrid[y0:y1, x0:x1]
            dist_sq = (xx - x) ** 2 + (yy - y) ** 2

            if is_head:
                # Head: bright with highlight
                circle = dist_sq <= r ** 2
                img[y0:y1, x0:x1][circle] = head_color
                # Add a small highlight dot for "eye" effect
                highlight_r = max(1, r // 3)
                highlight = dist_sq <= highlight_r ** 2
                bright = np.clip(head_color.astype(np.int16) + 60, 0, 255).astype(np.uint8)
                img[y0:y1, x0:x1][highlight] = bright
            else:
                # Body: alternating shade for segment ridges
                circle = dist_sq <= r ** 2
                if k % 2 == 0:
                    img[y0:y1, x0:x1][circle] = body_color
                else:
                    img[y0:y1, x0:x1][circle] = dark_body
