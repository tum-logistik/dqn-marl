import numpy as np
import torch
from environment.Gridworld import Gridworld
from IPython.display import clear_output
import random
from matplotlib import pylab as plt
from collections import deque
from environment.MarketEnv import MarketEnv
from common.properties import *

class DQNMARLNet():

    def __init__(self, state_size, action_space_size, output_size, 
                hidden_size = HIDDEN_SIZE, 
                gamma = GAMMA, 
                batch_size = BATCH_SIZE,
                loss_fn = DEFAULT_LOSS_FUNC,
                learning_rate = LEARNING_RATE,
                n_agents = 1):
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_size + action_space_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size,output_size)).to(device = devid)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.output_size = output_size
        self.state_dim = state_size
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.n_agents = n_agents
    
    def __call__(self, state):
        return self.model(state).to(device = devid)

    def extract_mini_batch(self, minibatch, state_dim):
        state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]).view(self.batch_size, state_dim).to(device = devid)
        action_batch = torch.tensor([a for (s1,a,r,s2,d) in minibatch]).type(torch.FloatTensor).to(device = devid)
        reward_batch = torch.tensor([r for (s1,a,r,s2,d) in minibatch]).type(torch.FloatTensor).to(device = devid)
        state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch]).view(self.batch_size, state_dim).to(device = devid)
        done_batch = torch.tensor([d for (s1,a,r,s2,d) in minibatch]).type(torch.FloatTensor).to(device = devid)
        return state1_batch, action_batch, reward_batch, state2_batch, done_batch
