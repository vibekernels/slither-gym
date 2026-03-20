"""Play the Slither environment manually with keyboard controls.

Controls:
    Left Arrow  / A  — turn left
    Right Arrow / D  — turn right
    Up Arrow    / W  — boost (costs length)
    (no key)         — go straight
    R                — restart after death
    ESC / Q          — quit
"""

import slither_gym  # noqa: F401 – registers the env
import gymnasium as gym


def main():
    import pygame

    env = gym.make("Slither-v0", render_mode="human", max_steps=999_999)
    obs, info = env.reset(seed=42)
    env.render()

    running = True
    total_reward = 0.0
    episode = 1

    print("=== Slither Gym — Human Play ===")
    print("Arrow keys or A/D to turn, R to restart, ESC/Q to quit")
    print(f"Episode {episode}")

    while running:
        action = 0  # straight by default

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    env.render()
                    total_reward = 0.0
                    episode += 1
                    print(f"\nEpisode {episode}")

        if not running:
            break

        keys = pygame.key.get_pressed()
        boosting = keys[pygame.K_UP] or keys[pygame.K_w]
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action = 1
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = 2
        if boosting:
            action += 3  # 3=boost straight, 4=boost left, 5=boost right

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()

        if terminated or truncated:
            reason = "died" if terminated else "time up"
            print(f"  {reason} | score={info['score']:.0f} "
                  f"length={info['length']} reward={total_reward:.1f}")
            # Wait for R to restart or Q/ESC to quit
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            waiting = False
                            running = False
                        elif event.key == pygame.K_r:
                            obs, info = env.reset()
                            env.render()
                            total_reward = 0.0
                            episode += 1
                            print(f"\nEpisode {episode}")
                            waiting = False
                pygame.time.wait(50)

    env.close()
    print("Goodbye!")


if __name__ == "__main__":
    main()
