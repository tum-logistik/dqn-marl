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

def build_one_hot(n, size):
    arr = np.zeros(size)
    arr[int(n)] = 1
    return arr

def run_marl(MARLAgent, 
            marketEnv = MarketEnv(action_size = ACTION_DIM), 
            epochs = EPOCHS, 
            batch_size = BATCH_SIZE,
            max_steps = MAX_STEPS,
            sync_freq = SYNC_FREQ,
            explore_epsilon = EXPLORE_EPSILON,
            agent_index = 0):

    marl_agent_list = []
    for a in range(marketEnv.n_agents):
        target_net = copy.deepcopy(MARLAgent.model)
        target_net.load_state_dict(MARLAgent.model.state_dict())
        marl_agent_list.append(target_net)
    
    episode_rewards = []
    episode_rewards_sum = []
    episode_rewards_agent = []

    avg_epoch_rewards = []
    avg_epoch_rewards_sum = []
    avg_epoch_rewards_agent = []

    losses = []
    losses_nash = []
    j = 0

    # s -> n -> a
    na_policy_dict = dict()
    for n in range(marketEnv.n_agents):
        na_policy_dict[n] = neutral_policy_dic

    # everyone same policy
    sna_policy_dict = dict()
    for s in range(marketEnv.state_space_size):
        key = repr(list(marketEnv.state_space[s]))
        sna_policy_dict[key] = na_policy_dict

    for i in range(epochs):
        state1_ = marketEnv.reset()
        state1 = torch.from_numpy(state1_).float().to(device = devid)
        
        status = 1
        mov = 0
        rewards = []
        
        while(status == 1): 
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

            epsilon_nash_arr, value_cur_policy, sna_policy_dict_iter = sim_anneal_optimize(marketEnv, sna_policy_dict, k_max = 9)
            epsilon_nash = np.sum(epsilon_nash_arr)

            exp = (state1, nn_index, joint_rewards, epsilon_nash, state2, done)
            
            replay.append(exp)
            state1 = state2
            
            rewards.append(joint_rewards)

            # print out
            print(agent_action_indices)
            print(joint_rewards)
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                Q1, Q2, X, Y, loss, loss_nash = MARLAgent.batch_update(minibatch, target_net, MARLAgent.state_dim)

                print(i, loss.item())
                print(i, loss_nash.item())
                clear_output(wait=True)
                
                # Q learning
                MARLAgent.optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                MARLAgent.optimizer.step()

                # Nash learning
                MARLAgent.optimizer.zero_grad()
                loss_nash.backward()
                losses_nash.append(loss_nash.item())
                MARLAgent.optimizer.step()

                if j % sync_freq == 0:
                    target_net.load_state_dict(MARLAgent.model.state_dict())

            if done or mov > max_steps:
                
                avg_episode_reward = np.mean(np.array(rewards))
                sum_episode_reward = np.sum(np.array(rewards))
                agent_episode_reward = np.array(rewards)[-1][agent_index] # single agent reward (agent of interest)
                episode_rewards.append(avg_episode_reward)
                episode_rewards_sum.append(sum_episode_reward)
                episode_rewards_agent.append(agent_episode_reward)
                status = 0
                mov = 0
        
        smoothing_factor = -50
        avg_epoch_rewards.append(np.mean(np.array(episode_rewards)[smoothing_factor:]))
        avg_epoch_rewards_sum.append(np.mean(np.array(episode_rewards_sum)[smoothing_factor:]))
        avg_epoch_rewards_agent.append(np.mean(np.array(episode_rewards_agent)[smoothing_factor:]))
    
    return np.array(losses), np.array(episode_rewards), np.array(avg_epoch_rewards), np.array(avg_epoch_rewards_sum), np.array(avg_epoch_rewards_agent)