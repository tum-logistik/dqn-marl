import numpy as np

class MarketEnv():

    def __init__(self, action_size, max_demand = 50, demand_slope = 0.5):
        self.max_inventory = 2500
        self.state_env_dim = 2
        
        self.inventory = self.max_inventory
        self.action_space = np.arange(0, action_size)
        self.action_size = action_size
        self.action_env_dim = len(self.action_space)

        random_price = np.random.uniform(0, action_size)
        self.current_state = np.array([self.inventory, random_price])

        self.max_demand = max_demand
        self.demand_slope = demand_slope

    # .seed
    def seed(self, seed):
        return None
    
    # .action_space

    # .reset()
    def reset(self):
        random_ref_price = np.random.uniform(0, self.action_size)
        self.inventory = self.max_inventory
        return np.array([self.inventory, random_ref_price])
    
    def get_random_price():
        return np.random.uniform(0, self.action_size)
        
    # .step(action)
    def step(self, action_index):
        set_price = self.action_space[action_index]
        previous_ref_price = self.current_state[1]

        demand_lambda = self.max_demand - self.demand_slope*set_price
        demand = np.floor(np.random.poisson(demand_lambda))
        
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
