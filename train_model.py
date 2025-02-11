from env import CustomEnv
from dqn_agent import DQNAgent
import numpy as np
import wandb

def train_model(episodes=1000, batch_size=32):
    # Initialize W&B
    wandb.init(project="dqn-agent")

    # Initialize the environment and agent
    env = CustomEnv(visible=False, world_size=125, hp=3,bonus_pixel_prop=0.15)
    state_size = (11, 11)  # Surrounding observation size
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Training loop
    for e in range(episodes):
        state = env.reset()
        state = env.get_surrounding_observation(radius=5)
        state = np.expand_dims(state, axis=0)
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = env.get_surrounding_observation(radius=5)
            next_state = np.expand_dims(next_state, axis=0)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # Log metrics to W&B
        wandb.log({"episode": e, "score": total_reward, "epsilon": agent.epsilon})

        print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2}")

        if e % 100 == 0:
            agent.save(f"dqn_agent_episode_{e}.pth")

    env.close()

if __name__ == "__main__":
    train_model()