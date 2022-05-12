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
BATCH_SIZE = copy.deepcopy(cfg['dqn_params']['batch_size'])

# DQN Iterative Parameters
EPOCHS = copy.deepcopy(cfg['dqn_params']['epochs'])
SYNC_FREQ = copy.deepcopy(cfg['dqn_params']['sync_freq'])
EXPLORE_EPSILON = copy.deepcopy(cfg['dqn_params']['explore_epsilon'])
MAX_STEPS = copy.deepcopy(cfg['dqn_params']['max_steps'])

MEM_SIZE = copy.deepcopy(cfg['dqn_params']['mem_size'])
replay = deque(maxlen = MEM_SIZE)

HIDDEN_SIZE = copy.deepcopy(cfg['dqn_params']['hidden_size'])
GAMMA = copy.deepcopy(cfg['dqn_params']['gamma'])
DEFAULT_LOSS_FUNC = torch.nn.HuberLoss()
LEARNING_RATE = copy.deepcopy(cfg['dqn_params']['learning_rate'])

# MDP Parameters
ACTION_DIM = copy.deepcopy(cfg['mdp_params']['action_dim'])
STATE_DIM = copy.deepcopy(cfg['mdp_params']['state_dim'])
N_AGENTS = copy.deepcopy(cfg['mdp_params']['n_agents'])

# MARKET PARAMETERS
MAX_DEMAND = copy.deepcopy(cfg['market_params']['max_demand'])
BETA_0 = copy.deepcopy(cfg['market_params']['beta0'])
BETA_1 = copy.deepcopy(cfg['market_params']['beta1'])
BETA_2 = copy.deepcopy(cfg['market_params']['beta2'])
MARKET_A = copy.deepcopy(cfg['market_params']['a'])

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
K_MAX_SA = copy.deepcopy(cfg['sa_params']['k_max'])

# MC PARAMETERS
MC_MAX_ITER = copy.deepcopy(cfg['mc_params']['mc_max_iter'])

# TURBO TRO PARAMETERS
TURBO_MAX_EVALS = copy.deepcopy(cfg['turbo_prams']['max_evals'])
TURBO_BATCH_SIZE = copy.deepcopy(cfg['turbo_prams']['batch_size'])
TURBO_N_INIT = copy.deepcopy(cfg['turbo_prams']['n_init'])

# Epsilon net minimization
EPSNET_OPTIM_STEPS = copy.deepcopy(cfg['eps_net_params']['optim_steps'])

PLOT_SMOOTHING_FACTOR = copy.deepcopy(cfg['plot_params']['smoothing_factor'])