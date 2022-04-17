import numpy as np
import copy
from opt.value_estimator import *
from common.properties import *
from turbo.turbo_1 import Turbo1


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

def turbo_optimize(env, sna_policy_dict, q_network, k_max = K_MAX_SA):
    
    sna_policy_dict_iter = copy.deepcopy(sna_policy_dict)
    nqe = NashQEstimator(env, q_network, sna_policy_dict_iter)
    value_initial_policy, _ = value_search_sample_policy_approx(env, sna_policy_dict_iter, q_network_input = q_network)
    # flat_sna_policy_dict_candidate = nqe.get_flattened_policy_dict(sna_policy_dict_iter)
    # value = nqe(flat_sna_policy_dict_candidate)
    devid_turbo = "cuda" if torch.cuda.is_available() else "cpu"

    turbo1 = Turbo1(
        f=nqe,  # Handle to objective function
        lb=nqe.lb,  # Numpy array specifying lower bounds
        ub=nqe.ub,  # Numpy array specifying upper bounds
        n_init = TURBO_N_INIT,  # Number of initial bounds from an Latin hypercube design
        max_evals = TURBO_MAX_EVALS,  # Maximum number of evaluations
        batch_size = TURBO_BATCH_SIZE,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device=devid_turbo,  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
    )

    turbo1.optimize()
    
    X = turbo1.X  # Evaluated points
    fX = turbo1.fX  # Observed values
    ind_best = np.argmin(fX)
    f_best, x_best = fX[ind_best], X[ind_best, :]

    # retrieve epsilon
    sna_policy_bbo = nqe.get_sna_policy_dict(x_best)
    value_vector_bbo, joint_action_vector_bbo = value_search_sample_policy_approx(env, sna_policy_bbo, q_network_input = q_network)
    epsilon = value_vector_bbo - value_initial_policy

    return epsilon, f_best, x_best, sna_policy_bbo
