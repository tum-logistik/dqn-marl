import numpy as np
import torch
from IPython.display import clear_output
import random
from collections import deque
import copy

if torch.cuda.is_available():
    devid = torch.device('cuda:0')
else:
    devid = torch.device('cpu')

# Shared Parameters
BATCH_SIZE = 250

# DQN Iterative Parameters
EPOCHS = 4000
EXPLORE_EPSILON = 0.1
MAX_STEPS = 25
SYNC_FREQ = 40
MEM_SIZE = 500
replay = deque(maxlen = MEM_SIZE)

# DQN Parameters
HIDDEN_SIZE = 200
GAMMA = 0.8
DEFAULT_LOSS_FUNC = torch.nn.MSELoss()
LEARNING_RATE = 1e-3

# MDP Parameters
ACTION_DIM = 301
STATE_DIM = 2