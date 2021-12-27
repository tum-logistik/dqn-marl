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

    def __init__(self, state_dim, output_size, 
                hidden_size = HIDDEN_SIZE, 
                gamma = GAMMA, 
                batch_size = BATCH_SIZE,
                loss_fn = DEFAULT_LOSS_FUNC,
                learning_rate = LEARNING_RATE):
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size,output_size)).to(device = devid)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.output_size = output_size
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.loss_fn = loss_fn
    
    def __call__(self, state):
        return self.model(state).to(device = devid)
    
    def batch_update(self, minibatch, target_net, state_dim):
        state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]).view(self.batch_size, state_dim).to(device = devid)
        action_batch = torch.tensor([a for (s1,a,r,s2,d) in minibatch]).type(torch.FloatTensor).to(device = devid)
        reward_batch = torch.tensor([r for (s1,a,r,s2,d) in minibatch]).type(torch.FloatTensor).to(device = devid)
        state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch]).view(self.batch_size, state_dim).to(device = devid)
        done_batch = torch.tensor([d for (s1,a,r,s2,d) in minibatch]).type(torch.FloatTensor).to(device = devid)

        # Q update
        Q1 = self(state1_batch).to(device = devid)
        with torch.no_grad():
            Q2 = target_net(state2_batch).to(device = devid)
        
        Y = reward_batch + self.gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
        X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(X, Y.detach())

        return Q1, Q2, X, Y, loss
