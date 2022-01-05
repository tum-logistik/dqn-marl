import numpy as np
import torch
from IPython.display import clear_output
import random
from collections import deque
from tests.test_gw import *
from environment.MarketEnv import MarketEnv
from common.properties import *
from dqn_net import DQNNet

# SA deepQ network will handle exploded MA net.
N_AGENTS = 2

VISIT_COUNTER = dict() # dic of (s, a) -> count
SAS_PROB_DIC = dict() # dic of (s, a) -> count

## State representation is a joing state of all inventories... 

class MARLAgent(DQNNet):

    def __init__(self, env, 
        n_agents = N_AGENTS):
        self.n_visit_dic = dict()
        self.s_trans_prob = dict()
        self.n_agents = n_agents

        for s in env.state_space:
            self.n_visit_dic[repr(s)] = 0 

        for s in env.state_space:
            for a in env.action_space:
                for s2 in env.state_space:    
                    self.s_trans_prob[repr([s, a, s2])] = 1 / env.state_space_size
        
        self.joint_action_size = env.action_size * env.n_agents

        super(MARLAgent, self).__init__(env.state_space_size, self.joint_action_size, 
            hidden_size = HIDDEN_SIZE, 
            gamma = GAMMA, 
            batch_size = BATCH_SIZE,
            loss_fn = DEFAULT_LOSS_FUNC,
            learning_rate = LEARNING_RATE)

    # n_agent starts from 0
    def prob_action(self, s, n_agent, 
            explore_epsilon = EXPLORE_EPSILON, 
            state_dim = STATE_DIM):
        # Epsilon greedy maximum of Q net
        q_values = self(s)
        
        if not torch.cuda.is_available():
            qval_np = q_values.data.numpy()
        else:
            qval_np = q_values.data.cpu().numpy()
        
        index = n_agent*ACTION_DIM
        q_slice = qval_np[index:index+ACTION_DIM] # slice of the Q(s, a) output belonging to the n_agent
        
        action_ind = np.argmax(q_slice)
        prob_output = np.ones(len(q_slice)) * (explore_epsilon / (state_dim - 1) )
        prob_output[action_ind] = 1 - explore_epsilon
        
        joint_action_prob = np.zeros(self.joint_action_size)
        joint_action_prob[index:index+ACTION_DIM] = prob_output

        return joint_action_prob
    
    def prob_state_trans(self, s, a, s_next, ):
        # 1 / |s| (to start)... update to: prob_state_trans() + (1 - prob_state_trans() )/(Num. visit_counter[s][a])
        return self.s_trans_prob[repr([s, a, s_next])]

    def state_q(self, s, n_agent):
        # sum over action space prob_action(s) * Q(s, a)
        prob_output = self.prob_action(s, n_agent)
        prob_sum = 0
        for p in prob_output:
            prob_sum += p
        return prob_sum
