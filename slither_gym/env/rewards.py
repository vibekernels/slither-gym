from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    food_eaten: float = 1.0
    kill_opponent: float = 5.0
    death_scale: float = 0.1       # death penalty = -death_scale * length
    survival_per_step: float = 0.0  # no flat survival bonus (avoids circling)
    boost_cost: float = -0.01


def compute_reward(events: dict, config: RewardConfig) -> float:
    reward = config.survival_per_step
    reward += events.get("food_eaten", 0.0) * config.food_eaten
    reward += events.get("killed_opponent", 0) * config.kill_opponent
    if events.get("died", False):
        length = events.get("length", 10)
        reward -= config.death_scale * length
    if events.get("boosting", False):
        reward += config.boost_cost
    # Subtract mass lost from boosting (same rate as food_eaten so recycling is net-zero)
    reward -= events.get("boost_pellets_dropped", 0) * config.food_eaten
    return reward
