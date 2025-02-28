import numpy as np
import torch
from IPython.display import clear_output
import random
from collections import deque
from tests.test_gw import *
from environment.MarketEnv import MarketEnv
from common.properties import *
from dqn.dqn_net import DQNNet
from common.hash_functions import *
from opt.bbo_sim_anneal import *
from opt.bbo_tro import *
import time

# from dataclasses import dataclass

class ResultObj:
    episode_rewards: np.array
    avg_episode_rewards: np.array
    avg_episode_rewards_sum: np.array
    avg_episode_rewards_agent: np.array
    avg_episode_rewards_all_agents: np.array
    losses: np.array
    losses_eps: np.array
    losses_nash: np.array
    sna_policy_dict_iter: dict
    mdp_env: any
    marl_params: dict
    state_tracker: np.array
    episode_actions: np.array
    config_params: any

def build_one_hot(n, size):
    arr = np.zeros(size)
    arr[int(n)] = 1
    return arr

def run_marl(MARLAgent, 
            marketEnv = MarketEnv(action_size = ACTION_DIM), 
            episodes = episodes, 
            batch_size = BATCH_SIZE,
            max_steps = MAX_STEPS,
            sync_freq = SYNC_FREQ,
            explore_epsilon = EXPLORE_EPSILON,
            agent_index = 0):

    # marl_agent_list = []
    # for a in range(marketEnv.n_agents):
    #     target_net = copy.deepcopy(MARLAgent.model)
    #     target_net.load_state_dict(MARLAgent.model.state_dict())
    #     marl_agent_list.append(target_net)
    
    target_net = copy.deepcopy(MARLAgent.model)
    target_net.load_state_dict(MARLAgent.model.state_dict())
    
    episode_rewards = []
    episode_rewards_sum = []
    episode_rewards_agent = []
    episode_rewards_all_agents = []
    avg_episode_rewards = []
    avg_episode_rewards_sum = []
    avg_episode_rewards_agent = []
    episode_actions = []
    avg_episode_rewards_all_agents = []

    losses = []
    losses_eps = []
    losses_nash = []

    j = 0

    # s -> n -> a
    na_policy_dict = dict()
    for n in range(marketEnv.n_agents):
        na_policy_dict[n] = copy.deepcopy(neutral_policy_dic)

    # everyone same policy
    sna_policy_dict = dict()
    for s in range(marketEnv.state_space_size):
        key = repr(list(marketEnv.state_space[s]))
        sna_policy_dict[key] = copy.deepcopy(na_policy_dict)

    state1_ = marketEnv.reset()
    state_tracker_episode = []

    start_time = time.time()
    last_time = start_time

    for i in range(episodes):
        
        print("#### Episode Number: " + str(i))
        cur_time = time.time()
        print("Time from begining: " + str(cur_time - start_time))
        print("Time per Episode: " + str(cur_time - last_time))
        last_time = cur_time 
        
        state1 = torch.from_numpy(state1_).float().to(device = devid)
        status = 1
        mov = 0
        rewards = []
        state_tracker = []
        joint_actions = []

        while(status == 1):

            state_tracker.append(marketEnv.current_state[-1]) # append the reference price (state)
            j += 1
            mov += 1
            
            if not torch.cuda.is_available():
                state1_np = state1.data.numpy()
            else:
                state1_np = state1.data.cpu().numpy()
            state1_np[-1] = int(np.clip(state1_np[-1], 0, marketEnv.action_size))

            # pick agent to play, loop over all agents
            agent_action_indices = np.zeros(MARLAgent.n_agents)
            agent_policies = np.zeros([MARLAgent.n_agents, MARLAgent.action_size])
            na_policy_dict = dict()
            for n in range(MARLAgent.n_agents):
                play_prob = MARLAgent.prob_action(state1) # passes through the q net (currently global)
                action_ind = np.random.choice(np.arange(0, marketEnv.action_size ), p=play_prob)
                
                # Execute action and upate state, and get reward + boolTerminal
                agent_action_indices[n] = action_ind
                agent_policies[n] = np.array(play_prob)
                na_policy_dict[n] = RangeMapDict(dict(zip(list(range_dict.keys()), agent_policies[n])))

            dic_key = repr(list(state1_np))
            sna_policy_dict[dic_key] = na_policy_dict
            
            state2_, joint_rewards, done, info_dic = marketEnv.joint_step(agent_action_indices)
            state2 = torch.from_numpy(state2_).float().to(device = devid)

            # create action long form
            nn_index = int(action_index_to_hash(agent_action_indices, step = marketEnv.action_size))
            joint_action_index = np.zeros(np.power(marketEnv.action_size, marketEnv.n_agents)).reshape(-1)
            joint_action_index[nn_index] = 1

            # Change to each agent, run an optimization!
            # epsilon_nash_arr, value_cur_policy, sna_policy_dict_iter = sim_anneal_optimize(marketEnv, sna_policy_dict, q_network_input = MARLAgent)
            epsilon_nash_arr, value_sum_bbo, policy_bbo, sna_policy_dict_iter = turbo_optimize_eps(marketEnv, sna_policy_dict, MARLAgent)

            # state-by-state maximization, single agent deviation -> epsilon

            # state1_index = list(sna_policy_dict.keys()).index(repr(list(state1_np)))
            # epsilon_nash = np.sum(epsilon_nash_arr) # optimize for sum or epsilons
            # epsilon_nash = epsilon_nash_arr[state1_index] # optimize for state's epsilon value
            na_policy_dict_epsmax = sna_policy_dict_iter[dic_key]
            na_policy_dict_epsmax_array = np.array([list(na_policy_dict_epsmax[k].range_dic.values()) for k in na_policy_dict_epsmax]).flatten()

            exp = (state1, nn_index, joint_rewards, epsilon_nash_arr, na_policy_dict_epsmax_array, state2, done)
            
            replay.append(exp)
            state1 = state2
            
            rewards.append(joint_rewards)
            joint_actions.append(agent_action_indices)

            # print out
            print(agent_action_indices)
            print(joint_rewards)
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size) # Using recent replays
                Q1, Q2, X, Y, loss, loss_eps, loss_nash = MARLAgent.batch_update(minibatch, target_net, MARLAgent.state_dim)

                print(i, loss.item())
                print(i, loss_nash.item())
                clear_output(wait=True)
                
                # Q learning
                MARLAgent.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                losses.append(loss.item())
                MARLAgent.optimizer.step()

                # Eps learning
                MARLAgent.eps_optimizer.zero_grad()
                loss_eps.backward(retain_graph=True)
                losses_eps.append(loss_eps.item())
                MARLAgent.eps_optimizer.step()
                
                if devid == torch.device('cuda:0'): torch.cuda.empty_cache()

                # Nash learning
                MARLAgent.nash_optimizer.zero_grad()
                loss_nash.backward(retain_graph=True)
                losses_nash.append(loss_nash.item())
                MARLAgent.nash_optimizer.step()

                if j % sync_freq == 0:
                    target_net.load_state_dict(MARLAgent.model.state_dict())
            
            if np.sum(done) == marketEnv.n_agents or mov > max_steps:
                # avg_episode_reward = np.mean(np.array(rewards))
                sum_episode_reward = np.sum(np.array(rewards))
                agent_episode_reward = np.array(rewards)[-1][agent_index] # single agent reward (agent of interest)
                all_agent_episode_reward = np.array(rewards)[-1]
                
                episode_rewards.append(np.array(rewards))
                episode_rewards_sum.append(sum_episode_reward)
                episode_rewards_agent.append(agent_episode_reward)
                episode_rewards_all_agents.append(np.array(all_agent_episode_reward))

                status = 0
                mov = 0

                state_tracker_episode.append(state_tracker)
        
        smoothing_factor = PLOT_SMOOTHING_FACTOR
        avg_episode_rewards.append(np.mean(np.array(episode_rewards)[smoothing_factor:]))
        avg_episode_rewards_sum.append(np.mean(np.array(episode_rewards_sum)[smoothing_factor:]))
        avg_episode_rewards_agent.append(np.mean(np.array(episode_rewards_agent)[smoothing_factor:]))
        avg_episode_rewards_all_agents.append(np.mean(np.array(episode_rewards_agent)[smoothing_factor:]))

        episode_actions.append(joint_actions)
    
    res = ResultObj()
    
    res.episode_rewards = np.array(episode_rewards)
    
    res.avg_episode_rewards = np.array(avg_episode_rewards)
    res.avg_episode_rewards_sum = np.array(avg_episode_rewards_sum)
    res.avg_episode_rewards_agent = np.array(avg_episode_rewards_agent)
    res.losses = np.array(losses)
    res.losses_eps = np.array(losses_eps)
    res.losses_nash = np.array(losses_nash)
    res.sna_policy_dict_iter = sna_policy_dict_iter
    res.mdp_env = marketEnv
    res.state_tracker = np.array(state_tracker_episode)
    res.episode_actions = np.array(episode_actions)
    res.avg_episode_rewards_all_agents = np.array(avg_episode_rewards_all_agents)

    marl_params = {
        "episodes": episodes,
        "explore_epsilon": explore_epsilon,
        "max_steps": max_steps,
        "sync_freq": sync_freq,
        "mem_size": MEM_SIZE,
        "turbo_max_evals": TURBO_MAX_EVALS,
        "turbo_batch_size": TURBO_MAX_EVALS,
        "turbo_n_init": TURBO_MAX_EVALS,
        "batch_size": BATCH_SIZE,
        "n_agents": N_AGENTS
    }

    config_params = {
        "config_params": cfg
    }

    res.marl_params = marl_params
    res.config_params = config_params

    return res


