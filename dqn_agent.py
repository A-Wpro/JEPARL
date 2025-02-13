import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import wandb

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-5):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque([1.0] * capacity, maxlen=capacity)  # Initialize with default priorities
        self.pos = 0
        self.size = 0

    def add(self, experience, priority):
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = max(priority, 1e-5)  # Ensure priorities are positive and non-zero
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        priorities = np.array(self.priorities)[:self.size]  # Only consider filled positions
        probabilities = priorities ** self.alpha
        probabilities_sum = probabilities.sum()

        if probabilities_sum <= 0.1 or np.isnan(probabilities_sum):
            # If sum is zero or NaN, fall back to uniform probabilities
            probabilities = np.ones_like(priorities) / len(priorities)
        else:
            probabilities /= probabilities_sum

        # Ensure no NaN values in probabilities
        if np.isnan(probabilities).any():
            probabilities = np.nan_to_num(probabilities, nan=1.0/len(probabilities), posinf=1.0/len(probabilities), neginf=1.0/len(probabilities))
        if sum(probabilities)!=1 and  probabilities.sum() <= 0.1 :
            print(probabilities.sum())
            probabilities = np.ones_like(priorities) / len(priorities)


        indices = np.random.choice(self.size, batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(priority, 1e-5)  # Ensure priorities are positive and non-zero

    def increase_beta(self):
        self.beta = min(1.0, self.beta + self.beta_increment)

    def __len__(self):
        return self.size

class DQNAgent:
    def __init__(self, state_size, action_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = PrioritizedReplayBuffer(capacity=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.05
        self.min_learning_rate = 0.000005
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99992)
        self.loss_fn = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler()

    def _build_model(self):
        state_size_flat = np.prod(self.state_size)
        model = nn.Sequential(
            nn.Linear(state_size_flat, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done, priority=1.0):
        experience = (state, action, reward, next_state, done)
        self.memory.add(experience, priority)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return  # Not enough experiences to sample a batch

        minibatch, weights, indices = self.memory.sample(batch_size)
        priorities = []
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

            priorities.append((loss.item() + 1e-5) ** self.memory.alpha)

        self.memory.update_priorities(indices, priorities)
        self.memory.increase_beta()
        self.scheduler.step()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], self.min_learning_rate)

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
            "loss": (self.loss if hasattr(self, 'loss') else 0 - 0) / (1 - 0)
        })
