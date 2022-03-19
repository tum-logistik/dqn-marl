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
BATCH_SIZE = 21

# DQN Iterative Parameters
EPOCHS = 27
EXPLORE_EPSILON = 0.1
MAX_STEPS = 25
SYNC_FREQ = 99
MEM_SIZE = 50
replay = deque(maxlen = MEM_SIZE)

# DQN Parameters
HIDDEN_SIZE = 200
GAMMA = 0.8
DEFAULT_LOSS_FUNC = torch.nn.HuberLoss()
LEARNING_RATE = 1e-3

# MDP Parameters
ACTION_DIM = 301
STATE_DIM = 2

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