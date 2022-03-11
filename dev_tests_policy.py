from opt.value_estimator import *
from environment.MarketEnv import *
from common.RangeMap import *
import numpy as np
from opt.bbo_sim_anneal import *


STATE_DIM = 2
ACTION_DIM = 11
BATCH_SIZE = 200

marketEnv = MarketEnv(action_size = ACTION_DIM, max_price = 100, max_inventory = 5, n_agents = 3)

range_dict = {
    (0, 10): 0.10,
    (10, 20): 0.10,
    (20, 30): 0.10,
    (30, 40): 0.10,
    (40, 50): 0.10,
    (50, 60): 0.10,
    (60, 70): 0.10,
    (70, 80): 0.10,
    (80, 90): 0.10,
    (90, 100): 0.10,
}

policy_dic = RangeMapDict(range_dict)

# s -> n -> a
na_policy_dict = dict()
for n in range(marketEnv.n_agents):
    na_policy_dict[n] = policy_dic

# everyone same policy
sna_policy_dict = dict()
for s in range(marketEnv.state_space_size):
    key = repr(list(marketEnv.state_space[s]))
    sna_policy_dict[key] = na_policy_dict

epsilon, value_cur_policy = sim_anneal_optimize(marketEnv, sna_policy_dict)