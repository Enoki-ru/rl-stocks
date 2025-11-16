import torch
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (torch.FloatTensor(np.array(state)),
                torch.FloatTensor(np.array(action)),
                torch.FloatTensor(np.array(reward)).unsqueeze(1),
                torch.FloatTensor(np.array(next_state)),
                torch.FloatTensor(np.array(done)).unsqueeze(1))

    def __len__(self) -> int:
        return len(self.buffer)