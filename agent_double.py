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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size
        
        # These are hyperparameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net and the target net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)
        
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        # Initialize a target network and synchronize it with the policy net
        self.target_net = DQN(action_size).to(device)
        self.update_target_net()

    def load_policy_net(self, path):
        self.policy_net = torch.load(path)

    # After some time interval, update the target network to be the same as the policy network
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # Explore: choose a random action
            action = random.randint(0, self.action_size - 1)
        else:
            # Exploit: choose the action with the highest Q-value
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = self.policy_net(state_tensor).argmax(dim=1).item()
        return action

    # Pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        # Prepare data for training
        history = np.stack(mini_batch[0], axis=0)
        states = torch.FloatTensor(history[:, :4, :, :] / 255.0).to(device)
        actions = torch.LongTensor(list(mini_batch[1])).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(list(mini_batch[2])).to(device)
        next_states = torch.FloatTensor(history[:, 1:, :, :] / 255.0).to(device)
        dones = torch.BoolTensor(list(mini_batch[3])).to(device)

        # Calculate Q(s_t, a_t) from the policy network
        state_action_values = self.policy_net(states).gather(1, actions)

        # Compute the Q value for the next state using the target network
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0  # Set Q values to 0 for terminal states

        # Compute the expected Q values
        expected_state_action_values = rewards + (self.discount_factor * next_state_values)

        # Compute the loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
