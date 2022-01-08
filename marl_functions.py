import numpy as np
import torch
from IPython.display import clear_output
import random
from collections import deque
from tests.test_gw import *
from environment.MarketEnv import MarketEnv
from common.properties import *
from dqn_net import DQNNet

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

    target_net = copy.deepcopy(MARLAgent.model)
    target_net.load_state_dict(MARLAgent.model.state_dict())

    episode_rewards = []
    episode_rewards_sum = []
    episode_rewards_agent = []

    avg_epoch_rewards = []
    avg_epoch_rewards_sum = []
    avg_epoch_rewards_agent = []

    losses = []
    j = 0

    for i in range(epochs):
        state1_ = marketEnv.reset()
        state1 = torch.from_numpy(state1_).float().to(device = devid)
        
        status = 1
        mov = 0
        rewards = []
        
        while(status == 1): 
            j += 1
            mov += 1
            
            # pick agent to play, loop over all agents
            agent_action_indices = np.zeros(MARLAgent.n_agents)
            for n in range(MARLAgent.n_agents):
                play_prob = MARLAgent.prob_action(state1, 0) # passes through the q net (currently global)

                if (random.random() < explore_epsilon):
                    action_ind = np.random.randint(0, int(MARLAgent.output_size / MARLAgent.n_agents) )
                else:
                    action_ind = np.argmax(play_prob)
                
                # Execute action and upate state, and get reward + boolTerminal
                agent_action_indices[n] = action_ind
            
            state2_, joint_rewards, done, info_dic = marketEnv.joint_step(agent_action_indices)
            state2 = torch.from_numpy(state2_).float().to(device = devid)

            # create action long form
            action_indice_longform = np.array([build_one_hot(x, MARLAgent.action_size) for x in agent_action_indices]).reshape(-1)

            exp = (state1, action_indice_longform, joint_rewards, state2, done)
            
            replay.append(exp)
            state1 = state2
            
            rewards.append(joint_rewards)

            # print out
            print(agent_action_indices)
            print(joint_rewards)
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                Q1, Q2, X, Y, loss = MARLAgent.batch_update_competitive(minibatch, target_net, MARLAgent.state_dim)

                print(i, loss.item())
                clear_output(wait=True)
                
                MARLAgent.optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                MARLAgent.optimizer.step()
                
                if j % sync_freq == 0:
                    target_net.load_state_dict(MARLAgent.model.state_dict())

            if done or mov > max_steps:
                
                avg_episode_reward = np.mean(np.array(rewards))
                sum_episode_reward = np.sum(np.array(rewards))
                agent_episode_reward = np.array(rewards)[-1][agent_index] # single agent reward (agent of interest)
                clear_output(wait=True)
                episode_rewards.append(avg_episode_reward)
                episode_rewards_sum.append(sum_episode_reward)
                episode_rewards_agent.append(agent_episode_reward)
                status = 0
                mov = 0
        
        smoothing_factor = -50
        avg_epoch_rewards.append(np.mean(np.array(episode_rewards)[smoothing_factor:]))
        avg_epoch_rewards_sum.append(np.mean(np.array(episode_rewards_sum)[smoothing_factor:]))
        avg_epoch_rewards_agent.append(np.mean(np.array(episode_rewards_agent)[smoothing_factor:] ))
    
    return np.array(losses), np.array(episode_rewards), np.array(avg_epoch_rewards), np.array(avg_epoch_rewards_sum), np.array(avg_epoch_rewards_agent)