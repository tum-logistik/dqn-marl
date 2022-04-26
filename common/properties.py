import numpy as np
import torch
from IPython.display import clear_output
import random
from collections import deque
import copy
from common.RangeMap import *

if torch.cuda.is_available():
    devid = torch.device('cuda:0')
else:
    devid = torch.device('cpu')

# Shared Parameters
BATCH_SIZE = 15

# DQN Iterative Parameters
EPOCHS = 60
EXPLORE_EPSILON = 0.05
MAX_STEPS = 30
SYNC_FREQ = 40
MEM_SIZE = 101333
replay = deque(maxlen = MEM_SIZE)

# DQN Parameters
HIDDEN_SIZE = 50
GAMMA = 0.8
DEFAULT_LOSS_FUNC = torch.nn.HuberLoss()
LEARNING_RATE = 1e-3

# MDP Parameters
ACTION_DIM = 10
STATE_DIM = 2
N_AGENTS = 3

# MARKET PARAMETERS
MAX_DEMAND = 3

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
K_MAX_SA = 21

# MC PARAMETERS
MC_MAX_ITER = 99

# TURBO TRO PARAMETERS
TURBO_MAX_EVALS = 13
TURBO_BATCH_SIZE = 4
TURBO_N_INIT = 3

# Epsilon net minimization
EPSNET_OPTIM_STEPS = 15