from env import CustomEnv
from dqn_agent import DQNAgent
import numpy as np

# Initialize the environment and agent
env = CustomEnv(visible=True, world_size=125, hp=3,bonus_pixel_prop=0.05,score_up=1000)
state_size = (11, 11)  # Surrounding observation size
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Load the trained model
agent.load("dqn_agent_100.pth")

# Run the environment with the trained agent
state = env.reset()
state = env.get_surrounding_observation(radius=5)
state = np.expand_dims(state, axis=0)
done = False

while not done:
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    next_state = env.get_surrounding_observation(radius=5)
    next_state = np.expand_dims(next_state, axis=0)
    state = next_state
    env.render()

env.close()
