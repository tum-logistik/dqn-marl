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
dummy_market = MarketEnv(action_size = ACTION_DIM)
STATE_SPACE_SIZE = dummy_market.state_space_size







# n_agent starts from 0
def prob_action(s, n_agent, dqn_model, explore_epsilon = EXPLORE_EPSILON, n_agents = N_AGENTS, action_dim = ACTION_DIM, state_dim = STATE_DIM):
    # Epsilon greedy maximum of Q net
    q_values = dqn_model(s)
    
    if not torch.cuda.is_available():
        qval_np = q_values.data.numpy()
    else:
        qval_np = q_values.data.cpu().numpy()
    
    index = n_agent*ACTION_DIM
    q_slice = qval_np[index:index+ACTION_DIM] # slice of the Q(s, a) output belonging to the n_agent
    
    action_ind = np.argmax(q_slice)
    prob_output = np.ones(len(q_slice)) * (explore_epsilon / (state_dim - 1) )
    prob_output[action_ind] = 1 - explore_epsilon
    
    # if (random.random() < explore_epsilon):
    #     # action_ind = np.random.randint(0, np.floor(dqn_model.output_size / n_agents))
    #     action_ind = np.random.randint(0, len(q_slice))
    # else:
    #     action_ind = np.argmax(q_slice)
    # subset q_values of agent
    return prob_output

def prob_state_trans(s_next, a, s, env, sas_prob_dic = SAS_PROB_DIC):
    # 1 / |s| (to start)... update to: prob_state_trans() + (1 - prob_state_trans() )/(Num. visit_counter[s][a])
    env_copy = copy.deepcopy(env)
    
    # return sas_prob_dic[]

def state_q():
    # sum over action space prob_action(s) * Q(s, a)
    return 1

