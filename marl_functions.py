import numpy as np
import torch
from environment.Gridworld import Gridworld
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

### Market Specific ###
dummy_market = MarketEnv(action_size = ACTION_DIM)
STATE_SPACE_SIZE = dummy_market.max_inventory * ACTION_DIM

# String inventory-price to denote state
for i in range(dummy_market.max_inventory):
    for a in range(ACTION_DIM):
        state_id = str(i) + "-" + str(a)
        VISIT_COUNTER[state_id] = 0
        SAS_PROB_DIC[state_id] = 1 / STATE_SPACE_SIZE
### End market specific ###

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

def prob_state_trans(s_next, a, s, sas_prob_dic = SAS_PROB_DIC):
    # 1 / |s| (to start)... update to: prob_state_trans() + (1 - prob_state_trans() )/(Num. visit_counter[s][a])
    
    
    # return sas_prob_dic[]

def state_q():
    # sum over action space prob_action(s) * Q(s, a)
    return 1

