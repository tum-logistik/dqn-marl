import numpy as np
import torch
from IPython.display import clear_output
import random
from matplotlib import pylab as plt
from collections import deque
import copy

if torch.cuda.is_available():
    devid = torch.device('cuda:0')
else:
    devid = torch.device('cpu')

mem_size = 1000
batch_size = 50
replay = deque(maxlen=mem_size)

epochs = 5000
losses = [] #A

max_moves = 50
h = 0
sync_freq = 500 #A

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3

gamma = 0.95
epsilon = 0.1
learning_rate = 1e-3
