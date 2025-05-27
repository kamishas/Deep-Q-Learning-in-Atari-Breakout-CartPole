from config import *
from collections import deque
import numpy as np
import random
import torch


class ReplayMemory(object):
    def __init__(self):
        self.memory = deque(maxlen=Memory_capacity)
    
    def push(self, history, action, reward, done):
        if torch.is_tensor(action):
            action = action.cpu().item()  # Convert tensor to scalar
        if torch.is_tensor(reward):
            reward = reward.cpu().item()  # Convert tensor to scalar
        self.memory.append((np.array(history), int(action), float(reward), bool(done)))


    def sample_mini_batch(self, frame):
        mini_batch = []
        if frame >= Memory_capacity:
            sample_range = Memory_capacity
        else:
            sample_range = frame

        # history size
        sample_range -= (HISTORY_SIZE + 1)

        idx_sample = random.sample(range(sample_range), batch_size)
        for i in idx_sample:
            sample = []
            for j in range(HISTORY_SIZE + 1):
                sample.append(self.memory[i + j])

            # Ensure consistent shapes for each element in the sample
            sample = np.array(sample, dtype=object)  # Use dtype=object for flexibility
            mini_batch.append((
                np.stack([s[0] for s in sample], axis=0),  # History frames
                sample[HISTORY_SIZE][1],  # Action
                sample[HISTORY_SIZE][2],  # Reward
                sample[HISTORY_SIZE][3]   # Done
            ))

        return mini_batch



    def __len__(self):
        return len(self.memory)


