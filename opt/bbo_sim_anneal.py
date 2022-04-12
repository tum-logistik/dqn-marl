import numpy as np
import copy
from opt.value_estimator import *
from common.properties import *
from turbo.turbo_1 import Turbo1

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

def turbo_optimize(env, sna_policy_dict, q_network_input = None, k_max = K_MAX_SA):
    
    sna_policy_dict_iter = copy.deepcopy(sna_policy_dict)
    nqe = NashQEstimator(env, q_network_input, sna_policy_dict_iter)
    flat_sna_policy_dict_candidate = nqe.get_flattened_policy_dict(sna_policy_dict_iter)
    # value = nqe(flat_sna_policy_dict_candidate)

    turbo1 = Turbo1(
        f=nqe,  # Handle to objective function
        lb=nqe.lb,  # Numpy array specifying lower bounds
        ub=nqe.ub,  # Numpy array specifying upper bounds
        n_init=20,  # Number of initial bounds from an Latin hypercube design
        max_evals = 200,  # Maximum number of evaluations
        batch_size=10,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
    )

    turbo1.optimize()
    
    X = turbo1.X  # Evaluated points
    fX = turbo1.fX  # Observed values
    ind_best = np.argmin(fX)
    f_best, x_best = fX[ind_best], X[ind_best, :]

    return f_best, x_best


def sim_anneal_optimize(env, sna_policy_dict, k_max = K_MAX_SA, q_func = None, q_network_input = None):
    sna_policy_dict_iter = copy.deepcopy(sna_policy_dict)
    value_initial_policy, _ = value_search_sample_policy_approx(env, sna_policy_dict_iter)
    value_cur_policy = value_initial_policy
    epsilon = np.ones(env.state_space_size)*np.Inf
    for k in range(k_max):
        T = temp_func(k, k_max)
        # sna_policy_dict_candidate = copy.deepcopy(sna_policy_dict_iter)
        sna_policy_dict_candidate = dict()
        for state in env.state_space:
            state_key = repr(list(state))
            sna_policy_dict_candidate[state_key] = dict()
            for n in range(env.n_agents):
                pol_dic = perturb_policy(sna_policy_dict_iter[state_key][n])
                sna_policy_dict_candidate[state_key][n] = copy.deepcopy(pol_dic)
        
        value_candidate_policy, _ = value_search_sample_policy_approx(env, sna_policy_dict_candidate, q_network = q_network_input)

        if accept_prob(value_cur_policy, value_candidate_policy, T) > np.random.uniform(0, 1):
            sna_policy_dict_iter = sna_policy_dict_candidate
            epsilon = value_candidate_policy - value_cur_policy
        # potentially add early termination
    
    return epsilon, value_cur_policy, sna_policy_dict_iter



def get_policydict_from_flatrep():
    return 1

def create_state_from_numeric_index():
    return 1
