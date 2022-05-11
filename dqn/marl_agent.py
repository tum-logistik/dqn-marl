import numpy as np
import torch
from IPython.display import clear_output
import random
from collections import deque
from tests.test_gw import *
from common.properties import *
from dqn.dqn_net import DQNNet

VISIT_COUNTER = dict() # dic of (s, a) -> count
SAS_PROB_DIC = dict() # dic of (s, a) -> count

## State representation is a joing state of all inventories... 

class MARLAgent(DQNNet):

    def __init__(self, env):
        self.n_visit_dic = dict()
        # self.s_trans_prob = dict()
        self.n_agents = env.n_agents
        self.action_size = env.action_size
        self.state_dim = self.n_agents + 1
        
        self.joint_action_size = env.action_size * env.n_agents
        
        comb_arg_ja = [env.action_space] * env.n_agents
        self.joint_action_space = np.array(np.meshgrid(*comb_arg_ja)).T.reshape(-1, self.n_agents)

        for s in env.state_space:
            self.n_visit_dic[repr(s)] = 0 

        # for s in env.state_space:
        #     for a in self.joint_action_space:
        #         for s2 in env.state_space:    
        #             self.s_trans_prob[repr([s, a, s2])] = 1 / env.state_space_size
        
        super(MARLAgent, self).__init__(env.state_env_dim, 
            np.power(self.action_size, self.n_agents), 
            hidden_size = HIDDEN_SIZE, 
            gamma = GAMMA, 
            batch_size = BATCH_SIZE,
            loss_fn = DEFAULT_LOSS_FUNC,
            learning_rate = LEARNING_RATE,
            n_agents = env.n_agents,
            action_space_size = len(env.action_space),
            state_space_size = env.state_space_size)

    # n_agent starts from 0
    def prob_action(self, s, n_agent = 0, explore_epsilon = EXPLORE_EPSILON):
        # Epsilon greedy maximum of Q net
        q_values = self(s)
        nash_pol = self.nash_policy_model(s)

        if not torch.cuda.is_available():
            # qval_np = q_values.data.numpy()
            nash_pol_np = nash_pol.data.numpy()
        else:
            # qval_np = q_values.data.cpu().numpy()
            nash_pol_np = nash_pol.data.cpu().numpy()
        
        index = n_agent * self.action_size
        policy_slice = nash_pol_np[index:index+self.action_size] # slice of the policy(s, a) output belonging to the n_agent
        norm_police_slice = [x/sum(policy_slice) for x in policy_slice]
        
        return norm_police_slice

        
    # def prob_state_trans(self, s, a, s_next):
    #     # 1 / |s| (to start)... update to: prob_state_trans() + (1 - prob_state_trans() )/(Num. visit_counter[s][a])
    #     return self.s_trans_prob[repr([s, a, s_next])]
    
    # def marl_prob_sas_update(self, s, a, s_next):
    #     old_psas = self.s_trans_prob[repr([s, a, s_next])]
    #     self.s_trans_prob[repr([s, a, s_next])] = old_psas + (1 - old_psas)/self.n_visit_dic[repr([s, a, s_next])]
    #     self.n_visit_dic[repr([s, a, s_next])] = self.n_visit_dic[repr([s, a, s_next])] + 1
    #     return self.s_trans_prob[repr([s, a, s_next])] 

    def state_q(self, s, n_agent):
        # sum over action space prob_action(s) * Q(s, a)
        prob_output = self.prob_action(s, n_agent)
        prob_sum = 0
        for p in prob_output:
            prob_sum += p
        return prob_sum

    
