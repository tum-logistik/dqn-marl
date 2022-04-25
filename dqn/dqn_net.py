import numpy as np
import torch
from environment.Gridworld import Gridworld
from IPython.display import clear_output
import random
from matplotlib import pylab as plt
from collections import deque
from environment.MarketEnv import MarketEnv
from common.properties import *

def reform_combination_n_probs(arr, n_agents = N_AGENTS, action_dim = ACTION_DIM):
    arr1 = arr.reshape(int(n_agents), int(action_dim))
    exploded_arr = np.array(np.meshgrid(*arr1 )).T.reshape(-1, int(n_agents))
    return exploded_arr

def batch_nprob_reform(batch, batch_size = BATCH_SIZE, n_agents = N_AGENTS, action_dim = ACTION_DIM):
    batch2 = batch.reshape(int(batch_size), int(n_agents*action_dim) )
    v1 = np.apply_along_axis(reform_combination_n_probs, 1, batch2)
    # v1 = reform_combination_n_probs_vec(batch)
    return v1

class DQNNet():

    def __init__(self, state_dim, output_size, 
                hidden_size = HIDDEN_SIZE, 
                gamma = GAMMA, 
                batch_size = BATCH_SIZE,
                loss_fn = DEFAULT_LOSS_FUNC,
                learning_rate = LEARNING_RATE,
                n_agents = 1,
                action_space_size = None, 
                state_space_size = None):
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)).to(device = devid)
        
        self.output_size = output_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.n_agents = n_agents
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        self.comb_action_space = int(np.power(self.action_space_size, self.n_agents))
        
        if self.n_agents > 1:
            self.nash_eps_net = self.nash_policy_model = torch.nn.Sequential(
                torch.nn.Linear(self.comb_action_space, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size*20),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size*20, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, self.state_space_size),
                torch.nn.Sigmoid()).to(device = devid)
            self.eps_optimizer = torch.optim.Adam(self.nash_eps_net.parameters(), lr=learning_rate)

            self.nash_policy_model = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size*20),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size*20, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, self.action_space_size * self.n_agents),
                torch.nn.Sigmoid()).to(device = devid)
            self.nash_optimizer = torch.optim.Adam(self.nash_policy_model.parameters(), lr=learning_rate/1e2)
    
    def __call__(self, state):
        return self.model(state).to(device = devid)

    def extract_mini_batch(self, minibatch, state_dim):
        state1_batch = torch.cat([s1 for (s1,a,r,e,ep,s2,d) in minibatch]).view(self.batch_size, state_dim).to(device = devid)
        action_batch = torch.tensor([a for (s1,a,r,e,ep,s2,d) in minibatch]).type(torch.FloatTensor).to(device = devid)
        reward_batch = torch.tensor([r for (s1,a,r,e,ep,s2,d)  in minibatch]).type(torch.FloatTensor).to(device = devid)
        epsilon_state_batch = torch.tensor([e for (s1,a,r,e,ep,s2,d) in minibatch]).type(torch.FloatTensor).to(device = devid) if self.n_agents > 1 else None
        nash_policy_batch = torch.tensor([ep for (s1,a,r,e,ep,s2,d) in minibatch]).type(torch.FloatTensor).to(device = devid) if self.n_agents > 1 else None
        state2_batch = torch.cat([s2 for (s1,a,r,e,ep,s2,d) in minibatch]).view(self.batch_size, state_dim).to(device = devid)
        done_batch = torch.tensor([d for (s1,a,r,e,ep,s2,d) in minibatch]).type(torch.FloatTensor).to(device = devid)
        return state1_batch, action_batch, reward_batch, epsilon_state_batch, nash_policy_batch, state2_batch, done_batch
    
    def batch_update(self, minibatch, target_net, state_dim, n_agent = 0): # cooperative update MA
        state1_batch, action_batch, reward_batch, epsilon_state_batch, nash_policy_batch, state2_batch, done_batch = self.extract_mini_batch(minibatch, state_dim)
        
        # Q update
        Q1 = self(state1_batch).to(device = devid)
        with torch.no_grad():
            Q2 = target_net(state2_batch).to(device = devid)
        
        if self.n_agents > 1:
            # Nash Policy Net: state -> policy
            nash_policy_pred = self.nash_policy_model(state1_batch)
            # zeros_tensor = torch.from_numpy(np.zeros(BATCH_SIZE)).float().to(device = devid)
            nash_policy_batch.requires_grad=True
            loss_nash = self.loss_fn(nash_policy_batch, nash_policy_pred)

            nash_policy_pred_np = nash_policy_pred.cpu().detach().numpy()
            nash_policy_pred_nagent_np = nash_policy_pred_np.reshape(int(BATCH_SIZE), int(self.n_agents), int(ACTION_DIM))

            nash_probs = batch_nprob_reform(nash_policy_pred_nagent_np)
            # indexed_nash_scalar = np.apply_over_axes(np.multiply, nash_probs, 1)

            indexed_nash_scalar = np.multiply.reduce(nash_probs, axis=(2))
            nash_scalar_torch = torch.from_numpy(indexed_nash_scalar)
            scaled_q_func = nash_scalar_torch * Q2

            # Q Function Net, state -> Nash Q
            max_Q2 = torch.max(scaled_q_func,dim=1)[0]
            Q_formula = reward_batch[:, n_agent] + self.gamma * ((1-done_batch[:, n_agent]) * max_Q2)
            Q_net = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

            # Epsilon Net: policy -> epsilon, optimize over the epsilon net
            eps_pred = self.nash_eps_net(scaled_q_func)
            epsilon_state_batch.requires_grad=True
            loss_eps = self.loss_fn(epsilon_state_batch, eps_pred)
            # Minimizing solution to Epsilon Net
                    
        else:
            max_Q2 = torch.max(Q2,dim=1)[0]
            Q_formula = reward_batch + self.gamma * ((1-done_batch) * max_Q2)
            Q_net = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        
        loss = self.loss_fn(Q_net, Q_formula.detach())
        
        if self.n_agents > 1:
            return Q1, Q2, Q_net, Q_formula, loss, loss_eps, loss_nash
        else:
            return Q1, Q2, Q_net, Q_formula, loss
