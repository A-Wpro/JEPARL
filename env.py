import gym
from gym import spaces
import numpy as np
import pygame

class CustomEnv(gym.Env):
    def __init__(self, agent_size=5, world_size=100, visible=False):
        super(CustomEnv, self).__init__()

        self.agent_size = agent_size
        self.world_size = world_size
        self.visible = visible

        # Define action and observation space
        # Actions: 0: left, 1: right, 2: up, 3: down, 4: idle
        self.action_space = spaces.Discrete(5)

        # Observation space: the entire world matrix
        self.observation_space = spaces.Box(low=0, high=2, shape=(world_size, world_size), dtype=np.int32)

        self.reset()

    def reset(self):
        # Initialize the world matrix
        self.world = np.ones((self.world_size, self.world_size), dtype=np.int32)

        # Place walls (non-walkable areas)
        self._place_walls()

        # Place damage zones
        self._place_damage_zones()

        # Initialize agent position and health
        self.agent_pos = np.array([self.world_size // 2, self.world_size // 2])
        self.agent_hp = 10
        self.score = 0

        if self.visible:
            self._init_render()

        return self._get_observation()

    def _place_walls(self):
        # Example: Place walls in a V shape
        for i in range(self.world_size // 2):
            self.world[i, i] = -1
            self.world[i, self.world_size - 1 - i] = -1

    def _place_damage_zones(self):
        # Randomly assign 15% of the pixels as damage zones
        total_pixels = self.world_size * self.world_size
        damage_pixels = int(0.01 * total_pixels)
        damage_indices = np.random.choice(total_pixels, damage_pixels, replace=False)
        damage_coords = np.unravel_index(damage_indices, (self.world_size, self.world_size))
        self.world[damage_coords] = 2

    def _get_observation(self):
        # Return the entire world matrix as the observation
        return self.world.copy()

    def step(self, action):
        # Define movement based on action
        move_map = {
            0: np.array([0, -1]),  # left
            1: np.array([0, 1]),   # right
            2: np.array([-1, 0]),  # up
            3: np.array([1, 0]),   # down
            4: np.array([0, 0])    # idle
        }

        new_pos = self.agent_pos + move_map[action]

        # Check for wall collision
        if self.world[new_pos[0], new_pos[1]] == -1:
            new_pos = self.agent_pos  # Stay in the same position

        # Update agent position
        self.agent_pos = new_pos

        # Check for damage zones
        if self.world[self.agent_pos[0], self.agent_pos[1]] == 2:
            self.agent_hp -= 1

        # Update score
        self.score += 1

        # Check if the agent is dead
        done = self.agent_hp <= 0

        # Return observation, reward, done, info
        observation = self._get_observation()
        reward = 1 if not done else -10  # Example reward structure
        info = {"score": self.score, "hp": self.agent_hp}

        return observation, reward, done, info

    def render(self, mode='human'):
        if not self.visible:
            return

        # Initialize rendering
        self.screen.fill((255, 255, 255))

        # Draw the world
        for y in range(self.world_size):
            for x in range(self.world_size):
                color = (0, 0, 0) if self.world[y, x] == -1 else (255, 0, 0) if self.world[y, x] == 2 else (255, 255, 255)
                pygame.draw.rect(self.screen, color, pygame.Rect(x * self.agent_size, y * self.agent_size, self.agent_size, self.agent_size))

        # Draw the agent
        pygame.draw.rect(self.screen, (0, 0, 255), pygame.Rect(self.agent_pos[1] * self.agent_size, self.agent_pos[0] * self.agent_size, self.agent_size, self.agent_size))

        pygame.display.flip()

    def _init_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.world_size * self.agent_size, self.world_size * self.agent_size))
        pygame.display.set_caption('Custom RL Environment')

    def close(self):
        if self.visible:
            pygame.quit()

env = CustomEnv(visible=True,world_size=100)

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    env.render()
    print(f"Score: {info['score']}, HP: {info['hp']}")

env.close()
