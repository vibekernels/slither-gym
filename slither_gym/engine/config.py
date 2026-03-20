from dataclasses import dataclass, field


@dataclass
class GameConfig:
    # Arena
    arena_radius: float = 500.0

    # Snake physics
    base_speed: float = 3.0
    boost_speed: float = 6.0
    turn_rate: float = 0.12          # radians per step
    initial_length: int = 10
    max_length: int = 500
    segment_spacing: float = 4.0     # distance between segments
    head_radius: float = 6.0
    body_radius: float = 5.0
    boost_mass_cost: float = 0.2     # length lost per step while boosting

    # Food
    max_food: int = 600
    food_radius: float = 4.0
    food_value: float = 1.0          # length gained per food
    initial_food: int = 200
    food_respawn_rate: int = 3       # new food per step (up to max)
    death_food_fraction: float = 0.8 # fraction of length dropped as food on death

    # NPCs
    num_npcs: int = 4
    npc_respawn_delay: int = 30      # steps before NPC respawns

    # Observation
    obs_size: int = 64
    viewport_radius: float = 120.0   # world units visible from head

    # Episode
    max_steps: int = 4000
