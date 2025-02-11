from env import CustomEnv
from dqn_agent import DQNAgent
import numpy as np
import wandb

env = CustomEnv(visible=False, world_size=125, hp=3,bonus_pixel_prop=0.15,score_up=10000)
state_size = (11, 11)  # Surrounding observation size
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
done = False
batch_size = 32

EPISODES = 1000
time_max = 100
scores = []  # List to store scores for each episode
wandb.init(project="dqn-agent")

for e in range(EPISODES):
    state = env.reset()
    state = env.get_surrounding_observation(radius=5)
    state = np.expand_dims(state, axis=0)
    print(f"Episode {e + 1}/{EPISODES}")
    episode_score = 0
    for time in range(time_max):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = env.get_surrounding_observation(radius=5)
        next_state = np.expand_dims(next_state, axis=0)
        reward = reward if not done else -10

        episode_score += reward

        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        
        if done or time==time_max-1:
            print(f"Episode: {e + 1}/{EPISODES}, Score: {env.score}, Epsilon: {agent.epsilon:.2}")
            wandb.log({"episode": e, "score": env.score, "epsilon": agent.epsilon})
            break
 
    # Log the score for this episode
    scores.append(episode_score)

# Save the trained model
agent.save("dqn_agent.pth")
env.close()
 