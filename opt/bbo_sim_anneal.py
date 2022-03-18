import numpy as np
import copy
from opt.value_estimator import *
from common.properties import *


def temp_func(k, k_max, const = 3.0):
    return const * np.exp(1 - ((k+1)/k_max))

def accept_prob(value_cur_policy, value_candidate_policy, T):
    if (value_cur_policy >= value_candidate_policy).all():
        acc_prob = 1
    else:
        acc_prob = np.exp(-(np.sum(value_candidate_policy) - np.sum(value_cur_policy))/T)    
    return acc_prob

def perturb_policy(policy_dic, st_dev = 0.03):
    alt_dic = copy.deepcopy(policy_dic.range_dic)
    noise_sum =  0
    
    for k in alt_dic.keys():
        noise = np.random.normal(0, st_dev, 1)[0]
        added_sum = np.clip(alt_dic[k] + noise, 0, 1.0) - alt_dic[k]
        alt_dic[k] = alt_dic[k] + added_sum
        noise_sum += added_sum
    
    max_tries = 10
    tries = 0
    while tries < max_tries:
        tries += 1
        rand_index = np.random.choice(len(alt_dic.keys()))
        rand_key = list(alt_dic.keys())[rand_index]
        
        if 0 <= alt_dic[rand_key] + noise_sum <= 1.0: 
            alt_dic[rand_key] = np.clip(alt_dic[rand_key] + noise_sum, 0, 1.0)
            
            break

    if np.sum(list(alt_dic.values())) < 0.999:
        missing_sum = 1.0 - np.sum(list(alt_dic.values()))
        while tries < max_tries:
            tries += 1
            rand_index = np.random.choice(len(alt_dic.keys()))
            rand_key = list(alt_dic.keys())[rand_index]
            
            if 0 <= alt_dic[rand_key] + missing_sum <= 1.0: 
                alt_dic[rand_key] = np.clip(alt_dic[rand_key] + missing_sum, 0, 1.0)
                
                break
    
    if np.sum(list(alt_dic.values())) > 1.001:
        missing_sum = np.sum(list(alt_dic.values())) - 1.0
        while tries < max_tries:
            tries += 1
            rand_index = np.random.choice(len(alt_dic.keys()))
            rand_key = list(alt_dic.keys())[rand_index]
            if 0 <= alt_dic[rand_key] - missing_sum <= 1.0: 
                alt_dic[rand_key] = np.clip(alt_dic[rand_key] - missing_sum, 0, 1.0)
                break
    
    if all(v >= 0.0 for v in alt_dic.values()) and np.sum(list(alt_dic.values())) < 1.001:
        return RangeMapDict(alt_dic)
    
    return policy_dic

def sim_anneal_optimize(env, sna_policy_dict, k_max = K_MAX_SA, q_func = None, q_network_input = None):
    sna_policy_dict_iter = copy.deepcopy(sna_policy_dict)
    value_initial_policy, _ = value_search_sample_policy_approx(env, sna_policy_dict_iter)
    value_cur_policy = value_initial_policy
    epsilon = np.ones(env.state_space_size)*np.Inf
    for k in range(k_max):
        T = temp_func(k, k_max)
        sna_policy_dict_candidate = copy.deepcopy(sna_policy_dict_iter)
        for state in env.state_space:
            state_key = repr(list(state))
            for n in range(env.n_agents):
                sna_policy_dict_candidate[state_key][n] = perturb_policy(sna_policy_dict_iter[state_key][n])
    
        value_candidate_policy, _ = value_search_sample_policy_approx(env, sna_policy_dict_candidate, q_network = q_network_input)

        if accept_prob(value_cur_policy, value_candidate_policy, T) > np.random.uniform(0, 1):
            sna_policy_dict_iter = sna_policy_dict_candidate
            epsilon = value_candidate_policy - value_cur_policy
        # potentially add early termination
    

    return epsilon, value_cur_policy, sna_policy_dict_iter