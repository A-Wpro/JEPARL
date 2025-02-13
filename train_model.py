from env import CustomEnv
from dqn_agent import DQNAgent
import numpy as np
import wandb

env = CustomEnv(visible=False, world_size=125, hp=1,bonus_pixel_prop=0.15,score_up=500)
state_size = (11, 11)  # Surrounding observation size
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
done = False
batch_size = 256

EPISODES = 5000
time_max = 100
wandb.init(project="dqn-agent")

for e in range(EPISODES):
    state = env.reset()
    state = env.get_surrounding_observation(radius=5)
    state = np.expand_dims(state, axis=0)
    print(f"Episode {e + 1}/{EPISODES}")
    for time in range(time_max):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = env.get_surrounding_observation(radius=5)
        next_state = np.expand_dims(next_state, axis=0)


         
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        
        if done or time==time_max-1:
            print(f"Episode: {e + 1}/{EPISODES}, Score: {reward}, time survived {time}, Epsilon: {agent.epsilon:.2}")
            wandb.log({"episode": e+1, "score": reward, "epsilon": agent.epsilon, "time survived": {time}})
            break
    if e == int(EPISODES*0.25) or e == int(EPISODES*0.50) or e == int(EPISODES*0.75):
        agent.save(f"dqn_agent_{e}.pth")
 
# Save the trained model
agent.save("dqn_agent.pth")
env.close()
 