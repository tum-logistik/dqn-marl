# from turtle import st
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
