import numpy as np
from marl_functions import *
from marl_agent import *

market_env = MarketEnv(action_size = 4, n_agents = 3, max_inventory = 4)
marl_agent = MARLAgent(market_env)

run_marl(marl_agent, marketEnv = market_env)


print("done")


