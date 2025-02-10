import gym
from gym import spaces
import numpy as np
import pygame

class CustomEnv(gym.Env):
    def __init__(self, agent_size=5, world_size=125, visible=False, dmg_pixel_prop=0.05, bonus_pixel_prop=0.01, hp=10,score_up = 1000):
        super(CustomEnv, self).__init__()

        self.agent_size = agent_size
        self.world_size = world_size
        self.visible = visible
        self.dmg_pixel_prop = dmg_pixel_prop
        self.bonus_pixel_prop = bonus_pixel_prop
        self.hp = hp
        self.score_up = score_up

        # Define action and observation space
        # Actions: 0: left, 1: right, 2: up, 3: down, 4: idle
        self.action_space = spaces.Discrete(5)

        # Observation space: the entire world matrix
        self.observation_space = spaces.Box(low=0, high=3, shape=(world_size, world_size), dtype=np.int32)

        self.reset()

    def reset(self):
        # Initialize the world matrix
        self.world = np.ones((self.world_size, self.world_size), dtype=np.int32)

        # Ensure no unintended values in the world matrix
        self._sanitize_world()

        # Place walls (non-walkable areas)
        self._place_walls()

        # Place damage zones
        self._place_damage_zones()

        # Place bonus zones
        self._place_bonus_zones()

        # Initialize agent position and health
        self.agent_pos = np.array([self.world_size // 2, self.world_size // 2])
        self.agent_hp = self.hp
        self.score = 0

        if self.visible:
            self._init_render()

        return self._get_observation()

    def _sanitize_world(self):
        # Reset any non-standard values in the world matrix
        self.world[self.world != 1] = 1

    def _place_walls(self):
        # Place walls around the world with a buffer zone
        buffer_size = 2
        self.world[:buffer_size, :] = -1
        self.world[-buffer_size:, :] = -1
        self.world[:, :buffer_size] = -1
        self.world[:, -buffer_size:] = -1

        # Place a V-shaped wall with a stair-like pattern
        v_size = self.world_size // 4
        for i in range(v_size):
            self.world[self.world_size // 2 - i, self.world_size // 2 - v_size // 2 + i] = -1
            self.world[self.world_size // 2 - i, self.world_size // 2 + v_size // 2 - i] = -1
            # Ensure no holes by filling adjacent cells
            if i < v_size - 1:
                self.world[self.world_size // 2 - i, self.world_size // 2 - v_size // 2 + i + 1] = -1
                self.world[self.world_size // 2 - i, self.world_size // 2 + v_size // 2 - i - 1] = -1
                self.world[self.world_size // 2 - i - 1, self.world_size // 2 - v_size // 2 + i] = -1
                self.world[self.world_size // 2 - i - 1, self.world_size // 2 + v_size // 2 - i] = -1

        # Place a round wall (circle) with a stair-like pattern
        center = (self.world_size // 4, self.world_size // 4)
        radius = self.world_size // 10
        for y in range(self.world_size):
            for x in range(self.world_size):
                if np.sqrt((x - center[0])**2 + (y - center[1])**2) <= radius:
                    self.world[y, x] = -1
                    # Ensure no holes by filling adjacent cells
                    if (x + y) % 2 == 0:
                        if x + 1 < self.world_size:
                            self.world[y, x + 1] = -1
                        if y + 1 < self.world_size:
                            self.world[y + 1, x] = -1

        # Place a small labyrinth with a stair-like pattern
        labyrinth_start = (self.world_size * 3 // 4, self.world_size * 3 // 4)
        self.world[labyrinth_start[0]-1:labyrinth_start[0]+2, labyrinth_start[1]-1] = -1
        self.world[labyrinth_start[0]-1, labyrinth_start[1]:labyrinth_start[1]+2] = -1
        self.world[labyrinth_start[0]+1, labyrinth_start[1]-1:labyrinth_start[1]+2] = -1
        self.world[labyrinth_start[0], labyrinth_start[1]+1] = -1
        # Ensure no holes by filling adjacent cells
        self.world[labyrinth_start[0]-1, labyrinth_start[1]+1] = -1
        self.world[labyrinth_start[0]+1, labyrinth_start[1]-1] = -1





    def _place_damage_zones(self):
        # Randomly assign damage zones, avoiding walls and buffer zones
        total_pixels = self.world_size * self.world_size
        damage_pixels = int(self.dmg_pixel_prop * total_pixels)
        damage_indices = np.random.choice(total_pixels, damage_pixels, replace=False)
        damage_coords = np.unravel_index(damage_indices, (self.world_size, self.world_size))

        for y, x in zip(damage_coords[0], damage_coords[1]):
            if self.world[y, x] == 1:  # Only place damage zones in walkable areas
                self.world[y, x] = 2

    def _place_bonus_zones(self):
        # Randomly assign bonus zones, avoiding walls and buffer zones
        total_pixels = self.world_size * self.world_size
        bonus_pixels = int(self.bonus_pixel_prop * total_pixels)
        bonus_indices = np.random.choice(total_pixels, bonus_pixels, replace=False)
        bonus_coords = np.unravel_index(bonus_indices, (self.world_size, self.world_size))

        for y, x in zip(bonus_coords[0], bonus_coords[1]):
            if self.world[y, x] == 1:  # Only place bonus zones in walkable areas
                self.world[y, x] = 3

    def _get_observation(self):
        # Return the entire world matrix as the observation
        return self.world.copy()
 
    def get_surrounding_observation(self, radius):
        y, x = self.agent_pos
        
        # Define bounds for submatrix extraction
        y_min, y_max = max(0, y - radius), min(self.world_size, y + radius + 1)
        x_min, x_max = max(0, x - radius), min(self.world_size, x + radius + 1)
        
        # Extract the submatrix
        submatrix = np.full((2 * radius + 1, 2 * radius + 1), -2, dtype=int)
        world_subsection = self.world[y_min:y_max, x_min:x_max]
        
        # Place extracted world values into submatrix
        submatrix[:world_subsection.shape[0], :world_subsection.shape[1]] = world_subsection
        
        # Ensure agent's position is always visible
        agent_relative_y, agent_relative_x = radius, radius
        submatrix[agent_relative_y, agent_relative_x] = 0
        
        # Ray-casting to determine blocked vision
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip agent's position
                
                steps = max(abs(dx), abs(dy))
                blocked = False
                for step in range(1, steps + 1):
                    check_y = y + (dy * step) // steps
                    check_x = x + (dx * step) // steps
                    
                    if 0 <= check_y < self.world_size and 0 <= check_x < self.world_size:
                        rel_y = check_y - y + radius
                        rel_x = check_x - x + radius
                        
                        if blocked:
                            submatrix[rel_y, rel_x] = -2  # Mark as not visible
                        elif self.world[check_y, check_x] == -1:  # Blocking object
                            blocked = True  # Start marking cells behind as -2
                            submatrix[rel_y, rel_x] = -1  # Keep the obstacle visible
                            continue  # Ensure all subsequent cells in this direction are blocked
                    else:
                        break  # Stop when out of bounds
        
        return submatrix


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

        # Check for bonus zones
        if self.world[self.agent_pos[0], self.agent_pos[1]] == 3:
            self.score += self.score_up
            self.world[self.agent_pos[0], self.agent_pos[1]] = 1  # Make the bonus zone disappear

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
                color = (0, 0, 0) if self.world[y, x] == -1 else (255, 0, 0) if self.world[y, x] == 2 else (0, 255, 0) if self.world[y, x] == 3 else (255, 255, 255)
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

 