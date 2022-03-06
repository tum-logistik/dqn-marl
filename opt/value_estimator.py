from common.RangeMap import *
import numpy as np
from opt.pseudo_q_generator import *

# policy represented by numerical increment mapping to probabilities

def policy_scalar(env, sna_policy_dict, joint_actions, state):
    
    prob_scalar = 1
    policy_dict = sna_policy_dict[state]

    for n in range(env.n_agents):
        prob_scalar *= policy_dict[n][int(joint_actions[n])]
    return prob_scalar


def value_search_sample_policy(env, sna_policy_dict, n_agent, max_iter = 999):
    
    value_vector = np.zeros(env.state_space_size)
    for s in range(env.state_space_size):
        max_val =  0
        i = 0
        while i < max_iter:
            rand_joint_actions = np.random.choice(env.action_space, env.n_agents)
            state_key = repr(list(env.state_space[s]))
            pol_scal = policy_scalar(env, sna_policy_dict, rand_joint_actions, state_key)
            candidate_val = pol_scal * convex_q_gen(s, rand_joint_actions)
            if candidate_val > max_val:
                max_val = candidate_val
            i += 1
        value_vector[s] = max_val
    
    return value_vector

