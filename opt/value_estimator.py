from turtle import st
from common.RangeMap import *
import numpy as np
from opt.pseudo_q_generator import *
import itertools
from common.hash_functions import *
import torch
from common.properties import *

# policy represented by numerical increment mapping to probabilities

def q_function_wrapper(s, joint_actions, q_network):
    # MARLAgent(s) MARLAgent(s)
    joint_actions_index = int(action_index_to_hash(joint_actions))
    q_vals = q_network(s)
    q_vals_np = q_vals.data.numpy() if not torch.cuda.is_available() else q_vals.data.cpu().numpy()

    return q_vals_np[joint_actions_index]

def policy_scalar(env, sna_policy_dict, joint_actions, state):
    
    prob_scalar = 1
    policy_dict = sna_policy_dict[state]

    for n in range(env.n_agents):
        n_action = int(joint_actions[n])
        prob_scalar *= policy_dict[n][n_action]
    return prob_scalar


def value_search_sample_policy_approx(env, sna_policy_dict, q_network_input = None, max_iter = MC_MAX_ITER):

    value_vector = np.zeros(env.state_space_size)
    joint_action_vector = np.zeros([env.state_space_size, env.n_agents])
    state_keys = list(env.state_space)

    for s in range(env.state_space_size):
        max_val =  0
        i = 0
        best_joint_action = None
        while i < max_iter:
            rand_joint_actions = np.random.choice(env.action_space, env.n_agents) # can be more efficient search over a
            state_key = repr(list(env.state_space[s]))
            pol_scal = policy_scalar(env, sna_policy_dict, rand_joint_actions, state_key)
            state_key_np = np.array(state_keys[s])
            state_key_torch = torch.from_numpy(state_key_np).float().to(device = devid)

            if q_network_input == None:
                candidate_val = pol_scal * convex_q_gen(s, rand_joint_actions)
            else:
                candidate_val = pol_scal * q_function_wrapper(state_key_torch, rand_joint_actions, q_network_input)
            
            if candidate_val > max_val:
                max_val = candidate_val
                best_joint_action = rand_joint_actions
            i += 1
        value_vector[s] = max_val
        joint_action_vector[s] = best_joint_action
    
    return value_vector, joint_action_vector


def value_search_sample_policy(env, sna_policy_dict, q_func_callback = convex_q_gen):

    value_vector = np.zeros(env.state_space_size)
    joint_action_vector = np.zeros([env.state_space_size, env.n_agents])

    for s in range(env.state_space_size):
        max_val =  0
        best_joint_action = None
        action_space_perm = np.array([list(x) for x in list(itertools.permutations(env.action_space))])
        for joint_actions in action_space_perm:
            joint_actions = np.random.choice(env.action_space, env.n_agents) # can be more efficient search over a
            state_key = repr(list(env.state_space[s]))
            pol_scal = policy_scalar(env, sna_policy_dict, joint_actions, state_key)
            candidate_val = pol_scal * q_func_callback(s, joint_actions)
            if candidate_val > max_val:
                max_val = candidate_val
                best_joint_action = joint_actions
        value_vector[s] = max_val
        joint_action_vector[s] = best_joint_action
    
    return value_vector, joint_action_vector

class NashQEstimator:
    def __init__(self, env, q_network, sna_policy_dict, 
                max_iter = MC_MAX_ITER, 
                dim = None, 
                n_agents = 3,
                action_dim = 10): 
        self.max_iter = max_iter
        self.n_agents = n_agents

        self.env = env
        self.q_network = q_network
        self.sna_policy_dict = sna_policy_dict # reverse array of percs to sna_policy_dict
        self.states = list(sna_policy_dict.keys())
        self.action_dim = action_dim

        if dim == None:
            self.dim = len(self.states) * self.n_agents * self.action_dim
        else:
            self.dim = 300
        self.lb = -5 * np.ones(self.dim)
        self.ub = 10 * np.ones(self.dim)
    
    def get_state_rep_from_index(self, index):
        state_index = int(index / (self.action_dim * self.n_agents))
        agent_perc_index = index % (self.action_dim * self.n_agents)
        agent_index = int(agent_perc_index / self.action_dim)
        perc_index = agent_perc_index % self.action_dim
        state_rep = self.states[state_index]
        return state_rep, agent_index, perc_index
    
    def get_flattened_policy_dict(self, sna_policy_dict_candidate):
        perc_list = []
        states = sna_policy_dict_candidate.keys()
        for state in sna_policy_dict_candidate.values():
            for ranges in state.values():
                range_dic = ranges.range_dic
                for percs in range_dic.values():
                    perc_list.append(percs)
        
        # flat_perc_list = [item for sublist in perc_list for item in sublist]
        return perc_list

    def get_sna_policy_dict(self, perc_array):
        # full rewrite, could be optimized
        sna_policy_dict_update = copy.deepcopy(self.sna_policy_dict)
        for i in range(0, len(perc_array)):
            state_rep, agent_index, perc_index = self.get_state_rep_from_index(i)
            state_agent_info = sna_policy_dict_update[state_rep][agent_index]
            keys = list(state_agent_info.range_dic.keys())
            key = keys[perc_index]

            # d1 = sna_policy_dict_update[state_rep][agent_index].range_dic[key]
            sna_policy_dict_update[state_rep][agent_index].range_dic[key] = perc_array[i]
            # d2 = sna_policy_dict_update[state_rep][agent_index].range_dic[key]

        for state_rep in self.states:
            for agent_index in range(self.n_agents):
                new_range_dic = sna_policy_dict_update[state_rep][agent_index].range_dic
                sna_policy_dict_update[state_rep][agent_index].range_dic_perc = RangeMapDict(new_range_dic).range_dic_perc
        return sna_policy_dict_update

    def __call__(self, perc_array):
        
        sna_policy_dict_update = self.get_sna_policy_dict(perc_array)
        value_vector, joint_action_vector = value_search_sample_policy_approx(self.env, sna_policy_dict_update, self.q_network)

        return -np.sum(value_vector)