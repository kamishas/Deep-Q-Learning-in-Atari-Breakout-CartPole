import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import DQN
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

# Define the device for computation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size

        # Define hyperparameters for training the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Initialize replay memory
        self.memory = ReplayMemory()  # Reverting to `self.memory`

        # Define the policy network
        self.policy_net = DQN(action_size).to(device)

        # Define optimizer and scheduler
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def load_policy_net(self, path):
        """Load the policy network from a file."""
        self.policy_net = torch.load(path)

    def get_action(self, state):
        """
        Select an action using the epsilon-greedy policy.
        """
        if np.random.random() <= self.epsilon:
            # Explore: select a random action
            action = torch.tensor([[random.randint(0, self.action_size - 1)]], device=device, dtype=torch.long)
        else:
            # Exploit: select the best action based on the current policy
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).to(device)
                action = self.policy_net(state_tensor).argmax(dim=1).view(1, 1)
        return action

    def train_policy_net(self, frame):
        """
        Train the policy network using a batch of experiences from memory.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        # Sample a batch of experiences from replay memory
        mini_batch = self.memory.sample_mini_batch(frame)

        # Extract elements from the batch
        history = np.stack([data[0] for data in mini_batch], axis=0)
        states = torch.FloatTensor(history[:, :4, :, :] / 255.0).to(device)
        actions = torch.LongTensor([data[1] for data in mini_batch]).unsqueeze(1).to(device)
        rewards = torch.FloatTensor([data[2] for data in mini_batch]).to(device)
        next_states = torch.FloatTensor(history[:, 1:, :, :] / 255.0).to(device)
        dones = torch.BoolTensor([data[3] for data in mini_batch]).to(device)

        # Calculate Q(s_t, a_t)
        state_action_values = self.policy_net(states).gather(1, actions)

        # Calculate Q(s_t+1, a_t+1) for the next states
        with torch.no_grad():
            next_state_values = self.policy_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0

        # Compute the expected Q values
        expected_values = rewards + (self.discount_factor * next_state_values)

        # Compute loss
        loss = nn.SmoothL1Loss()(state_action_values, expected_values.unsqueeze(1))

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
