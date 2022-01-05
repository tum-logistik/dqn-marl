import numpy as np
import itertools

# np.random.bit_generator = np.random._bit_generator

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
    
    # state is defined as inventories of each agent + 
    def reset(self):
        random_ref_price = np.random.uniform(0, self.action_size)
        self.inventory = self.max_inventory
        return np.array([self.inventory] * self.n_agents + [random_ref_price])
    
    def get_random_price(self):
        return np.random.uniform(0, self.action_size)
        
    # .step(action)
    # Takes price index as step argument
    def step(self, action_index, n_agents = 1):
        
        # previous_ref_price = self.current_state[-1]
        set_price = self.action_space[action_index]
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
    
    def joint_step(self, action_indices):
        
        # previous_ref_price = self.current_state[-1]
        action_values = action_indices # special case
        selected_joint_act_index = int(np.min(action_values)) # minimum or average or any other function, current or previous time
        set_price = self.action_space[selected_joint_act_index]
        demand_lambda = self.max_demand - self.demand_slope*set_price

        clipped_lambda = np.max([demand_lambda, 0])
        demand = np.floor(np.random.poisson(clipped_lambda))        
        
        # different rewards for each agent, based on realized demand
        auction_counts = self.auction_system(action_values, demand)

        # reward = set_price * demand
        actionable_actions = auction_counts # available inventory
        inventories = self.current_state[0:self.n_agents] - auction_counts
        for i in range(len(inventories)):
            if inventories[i] < 0:
                actionable_actions[i] = self.current_state[0:self.n_agents][i] # sold out, take previous inventory
                inventories[i] = 0
        
        rewards = np.multiply(actionable_actions, action_values+1) # limit min sale price to 1

        # state update, reminder last index = set price
        self.current_state[self.n_agents] = set_price
        if inventories.any() > 0:
            inventory_limit = False
            self.current_state[0:self.n_agents] = inventories
            
        else:
            inventory_limit = True
            self.current_state[0:self.n_agents] = self.max_inventory
        
        next_state = self.current_state
        
        return next_state, rewards, inventory_limit, dict()

    def auction_system(self, agent_actions, demand):

        # assume indices is price value
        # new_ref_price = np.min(agent_actions)
        win_probs = np.exp(agent_actions)/np.sum(np.exp(agent_actions))
        agent_ids = np.arange(0, len(agent_actions))
        auction_win_agent = np.zeros(len(agent_actions))

        for d in range(int(demand)):
            vi = np.random.choice(agent_ids, 1, p=win_probs, replace=False)
            auction_win_agent[vi] += 1
        
        return auction_win_agent
            



    def price_translate_from_index(self, i):
        return float(i)
