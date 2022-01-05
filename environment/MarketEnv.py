import numpy as np
import itertools

class MarketEnv():

    def __init__(self, action_size, max_demand = 50, demand_slope = 0.5, n_agents = 1, max_inventory = 2500):
        # MARL parameters
        self.visit_counter = dict()
        self.n_agents = n_agents

        self.max_inventory = max_inventory
        self.state_env_dim = n_agents + 1
        
        self.inventory = self.max_inventory
        self.inventory_space_single = np.arange(0, max_inventory)
        self.action_space = np.arange(0, action_size)
        self.action_space_joint = np.arange(0, action_size * n_agents)
        self.action_size = action_size
        self.action_env_dim = len(self.action_space)

        random_price = np.random.uniform(0, action_size)
        self.current_state = np.array([self.inventory]*self.n_agents + [random_price])

        self.max_demand = max_demand
        self.demand_slope = demand_slope
        # self.state_space_size = self.max_inventory * action_size

        # State space
        comb_arg = [self.inventory_space_single] * self.n_agents + [self.action_space] # includes reference price
        self.state_space = np.array(np.meshgrid(*comb_arg)).T.reshape(-1, self.n_agents + 1)  #.T.reshape(-1, self.n_agents)
            
        
        self.state_space_size = len(self.state_space)
        

    # def make_combination(args):

    # .seed
    def seed(self, seed):
        return None
    
    # .reset()
    def reset(self):
        random_ref_price = np.random.uniform(0, self.action_size)
        self.inventory = self.max_inventory
        return np.array([self.inventory, random_ref_price])
    
    def get_random_price(self):
        return np.random.uniform(0, self.action_size)
        
    # .step(action)
    # Takes price index as step argument
    def step(self, action_index):
        set_price = self.action_space[action_index]
        previous_ref_price = self.current_state[1]

        demand_lambda = self.max_demand - self.demand_slope*set_price
        clipped_lambda = np.max([demand_lambda, 0])
        demand = np.floor(np.random.poisson(clipped_lambda))
        
        # reward = ((previous_ref_price + set_price)/2) * demand
        reward = set_price * demand
        self.current_state[0] = self.current_state[0] - demand

        inventory = self.current_state[0] 
        next_state = np.array([inventory , set_price])

        if inventory  > 0:
            inventory_limit = False
        else:
            inventory_limit = True
            self.current_state[0] = self.max_inventory
        
        return next_state, reward, inventory_limit, dict()

    def price_translate_from_index(self, i):
        return float(i)
