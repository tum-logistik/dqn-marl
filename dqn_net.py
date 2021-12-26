import numpy as np
import torch
from environment.Gridworld import Gridworld
from IPython.display import clear_output
import random
from matplotlib import pylab as plt
from collections import deque
from environment.MarketEnv import MarketEnv
from common.properties import *

class DQNNet():

    def __init__(self, state_dim = 2, hidden_size = 150, output_size = 101):
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size,output_size)).to(device = devid)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def __call__(self, state):
        return self.model(state).to(device = devid)
    
    

# class DQN():