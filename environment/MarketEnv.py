from re import A
import numpy as np

class MarketEnv():
    def __init__(self, 
        action_size, 
        max_price = None, 
        max_demand = 50, 
        
        n_agents = 1, 
        max_inventory = 2500, 
        max_belief = 1, 
        beta0 = 25,
        beta1 = -1.1, 
        beta2 = -2, 
        a = 0.1):

        if max_price is None:
            max_price = action_size # increment of 1
        
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.a = a
        self.demand_slope = self.beta0 +  self.beta1 + self.beta2

        self.n_agents = n_agents
        self.state_env_dim = n_agents + 1
        
        self.max_inventory = max_inventory
        self.inventory = self.max_inventory
        self.inventory_space_single = np.arange(0, max_inventory+1)

        self.action_space = np.arange(0, action_size) * (max_price / action_size)
        self.action_space_joint = np.arange(0, action_size * n_agents)
        self.action_size = action_size
        self.action_env_dim = len(self.action_space)

        random_price = np.random.uniform(0, max_price)
        self.current_state = np.array([self.inventory]*self.n_agents + [random_price])

        self.max_demand = max_demand

        # State space
        comb_arg = [self.inventory_space_single] * self.n_agents + [self.action_space] # includes reference price
        self.state_space = np.array(np.meshgrid(*comb_arg)).T.reshape(-1, self.n_agents + 1)  #.T.reshape(-1, self.n_agents)
        self.state_space_size = len(self.state_space)

        self.max_belief = max_belief

    def reset(self):
        random_ref_price = np.random.uniform(0, self.action_size)
        self.inventory = self.max_inventory
        return np.array([self.inventory] * self.n_agents + [random_ref_price])
    
    def step(self, action_index):
        
        # previous_ref_price = self.current_state[-1]
        set_price = self.action_space[action_index] # needs edit

        ref_price = self.current_state[-1]
        y_intercept = -self.beta2 * ref_price

        demand_lambda = y_intercept - self.demand_slope*set_price
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
    
    def joint_step(self, action_indices, n_agent_index = 0):
        
        # previous_ref_price = self.current_state[-1]
        action_values = action_indices # special case
        # selected_joint_act_index = int(np.max(action_values)) # refernce price: **minimum or average or any other function**, current or previous time
        # set_price = self.action_space[selected_joint_act_index]
        set_price = self.current_state[-1]
        y_intercept = -self.beta2 * set_price

        demand_lambda = 0
        for a in range(self.n_agents):
            demand_lambda += y_intercept - self.demand_slope*self.action_space[int(action_indices[a])]

        clipped_lambda = np.max([demand_lambda, 0])
        demand = np.floor(np.random.poisson(clipped_lambda))        
        
        # different rewards for each agent, based on realized demand
        auction_counts = self.auction_system(action_values, demand)

        # reward = set_price * demand
        if self.max_inventory > 0:
            actionable_actions = auction_counts # available inventory
            inventories = self.current_state[0:self.n_agents] - auction_counts
            for i in range(len(inventories)):
                if inventories[i] < 0:
                    actionable_actions[i] = self.current_state[0:self.n_agents][i] # sold out, take previous inventory
                    inventories[i] = 0
        
            # limit min sale price to 1

            # state update, reminder last index = set price
            self.current_state[self.n_agents] = set_price
            if inventories.any() > 0:
                inventory_limit = False
                self.current_state[0:self.n_agents] = inventories
                
            else:
                inventory_limit = True
                self.current_state[0:self.n_agents] = self.max_inventory
        else:
            actionable_actions = np.ones(self.n_agents) # available inventory
            inventory_limit = 0
        
        rewards = np.multiply(actionable_actions, action_values+1)
        new_ref_price = np.mean(action_values+1)
        self.current_state[-1] = new_ref_price
        next_state = self.current_state
        
        return next_state, rewards, [inventory_limit] * self.n_agents, dict()
    
    def map_belief(self, x):
        x_intercept = self.max_demand / self.demand_slope
        belief_slope = self.max_belief / x_intercept
        return self.max_belief - belief_slope*x
    
    def auction_system(self, agent_actions, demand):

        # assume indices is price value
        # new_ref_price = np.min(agent_actions)
        
        belief_func_vec = np.vectorize(self.map_belief)
        agent_beliefs = belief_func_vec(agent_actions)

        win_probs = np.exp(agent_beliefs)/np.sum(np.exp(agent_beliefs))
        agent_ids = np.arange(0, len(agent_actions))
        auction_win_agent = np.zeros(len(agent_actions))

        for d in range(int(demand)):
            vi = np.random.choice(agent_ids, 1, p=win_probs, replace=False)
            auction_win_agent[vi] += 1
        
        return auction_win_agent