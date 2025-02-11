import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import wandb
class DQNAgent:
    def __init__(self, state_size, action_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1  # Adjusted learning rate
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)  # Decaying learning rate
        self.loss_fn = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler()

    def _build_model(self):
        # Flatten the state size
        state_size_flat = np.prod(self.state_size)
        model = nn.Sequential(
            nn.Linear(state_size_flat, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().numpy())
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0).to(self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32).flatten().unsqueeze(0).to(self.device)
            action = torch.LongTensor([action]).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            done = torch.FloatTensor([done]).to(self.device)

            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.target_model(next_state))

            target_f = self.model(state).clone()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = self.loss_fn(self.model(state), target_f)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.scheduler.step()  # Update learning rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.loss = loss.item()
        self.log_variables()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def log_variables(self):
        wandb.log({
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "epsilon": self.epsilon,
            "loss": self.loss if hasattr(self, 'loss') else 0
        })