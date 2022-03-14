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
                learning_rate = LEARNING_RATE,
                n_agents = 1,
                action_space_size = None):
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size,output_size)).to(device = devid)
        
        self.output_size = output_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.n_agents = n_agents
        self.action_space_size = action_space_size

        if self.n_agents > 1:
            self.nash_policy_model = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, self.n_agents * self.action_space_size ),
                torch.nn.Sigmoid()).to(device = devid)
    
    def __call__(self, state):
        return self.model(state).to(device = devid)

    def extract_mini_batch(self, minibatch, state_dim):
        state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]).view(self.batch_size, state_dim).to(device = devid)
        action_batch = torch.tensor([a for (s1,a,r,s2,d) in minibatch]).type(torch.FloatTensor).to(device = devid)
        reward_batch = torch.tensor([r for (s1,a,r,s2,d) in minibatch]).type(torch.FloatTensor).to(device = devid)
        state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch]).view(self.batch_size, state_dim).to(device = devid)
        done_batch = torch.tensor([d for (s1,a,r,s2,d) in minibatch]).type(torch.FloatTensor).to(device = devid)
        return state1_batch, action_batch, reward_batch, state2_batch, done_batch
    
    def batch_update(self, minibatch, target_net, state_dim, n_agent = 0): # cooperative update MA

        state1_batch, action_batch, reward_batch, state2_batch, done_batch = self.extract_mini_batch(minibatch, state_dim)
        
        # Q update
        Q1 = self(state1_batch).to(device = devid)
        with torch.no_grad():
            Q2 = target_net(state2_batch).to(device = devid)
        
        if self.n_agents > 1:
            max_Q2 = torch.max(Q2,dim=1)[0]
            Q_formula = reward_batch[:, n_agent] + self.gamma * ((1-done_batch[:, n_agent]) * max_Q2)
            Q_net = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        else:
            max_Q2 = torch.max(Q2,dim=1)[0]
            Q_formula = reward_batch + self.gamma * ((1-done_batch) * max_Q2)
            Q_net = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        
        loss = self.loss_fn(Q_net, Q_formula.detach())

        return Q1, Q2, Q_net, Q_formula, loss
