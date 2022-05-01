import numpy as np
import torch
from IPython.display import clear_output
import random
from collections import deque
import copy
from common.RangeMap import *
import yaml

with open("config/config_file.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

if torch.cuda.is_available():
    devid = torch.device('cuda:0')
else:
    devid = torch.device('cpu')

# Shared Parameters
BATCH_SIZE = cfg['dqn_params']['batch_size']

# DQN Iterative Parameters
EPOCHS = cfg['dqn_params']['epochs']
SYNC_FREQ = cfg['dqn_params']['sync_freq']
EXPLORE_EPSILON = cfg['dqn_params']['explore_epsilon']
MAX_STEPS = cfg['dqn_params']['max_steps']

MEM_SIZE = cfg['dqn_params']['mem_size']
replay = deque(maxlen = MEM_SIZE)

HIDDEN_SIZE = cfg['dqn_params']['hidden_size']
GAMMA = cfg['dqn_params']['gamma']
DEFAULT_LOSS_FUNC = torch.nn.HuberLoss()
LEARNING_RATE = cfg['dqn_params']['learning_rate']

# MDP Parameters
ACTION_DIM = cfg['mdp_params']['action_dim']
STATE_DIM = cfg['mdp_params']['state_dim']
N_AGENTS = cfg['mdp_params']['n_agents']

# MARKET PARAMETERS
MAX_DEMAND = cfg['market_params']['max_demand']
BETA_0 = cfg['market_params']['beta0']
BETA_1 = cfg['market_params']['beta1']
BETA_2 = cfg['market_params']['beta2']
MARKET_A = cfg['market_params']['a']

range_dict = {
    (0, 1): 0.10,
    (1, 2): 0.10,
    (2, 3): 0.10,
    (3, 4): 0.10,
    (4, 5): 0.10,
    (5, 6): 0.10,
    (6, 7): 0.10,
    (7, 8): 0.10,
    (8, 9): 0.10,
    (9, 10): 0.10,
}

neutral_policy_dic = RangeMapDict(range_dict)

# SA PARAMETERS
K_MAX_SA = cfg['sa_params']['k_max']

# MC PARAMETERS
MC_MAX_ITER = cfg['mc_params']['mc_max_iter']

# TURBO TRO PARAMETERS
TURBO_MAX_EVALS = cfg['turbo_prams']['max_evals']
TURBO_BATCH_SIZE = cfg['turbo_prams']['batch_size']
TURBO_N_INIT = cfg['turbo_prams']['n_init']

# Epsilon net minimization
EPSNET_OPTIM_STEPS = cfg['eps_net_params']['optim_steps']