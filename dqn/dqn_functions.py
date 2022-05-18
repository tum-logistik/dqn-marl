import numpy as np
import torch
from environment.Gridworld import Gridworld
from IPython.display import clear_output
import random
from matplotlib import pylab as plt
from collections import deque
from tests.test_gw import *
from environment.MarketEnv import MarketEnv
from common.properties import *
from dqn.dqn_net import DQNNet

DQNModel = DQNNet(state_dim = STATE_DIM, output_size = ACTION_DIM)

def run_dqn(DQNModel, 
            marketEnv = MarketEnv(action_size = ACTION_DIM), 
            episodes = episodes, 
            batch_size = BATCH_SIZE,
            max_steps = MAX_STEPS,
            sync_freq = SYNC_FREQ,
            explore_epsilon = EXPLORE_EPSILON):

    target_net = copy.deepcopy(DQNModel.model)
    target_net.load_state_dict(DQNModel.model.state_dict())

    episode_rewards = [] 
    average_episode_rewards = []
    losses = []
    j = 0

    for i in range(episodes):
        state1_ = marketEnv.reset()
        state1 = torch.from_numpy(state1_).float().to(device = devid)
        status = 1
        mov = 0
        rewards = []
        
        while(status == 1): 
            j+=1
            mov += 1
            qval = DQNModel(state1)
            
            if not torch.cuda.is_available():
                qval_ = qval.data.numpy()
            else:
                qval_ = qval.data.cpu().numpy()
            
            if (random.random() < explore_epsilon):
                action_ind = np.random.randint(0, DQNModel.output_size)
            else:
                action_ind = np.argmax(qval_)
            
            # Execute action and upate state, and get reward + boolTerminal
            action = action_ind
            marketEnv.step(action)
            state2_, reward, done, info_dic = marketEnv.step(action)
            state2 = torch.from_numpy(state2_).float().to(device = devid)
            exp = (state1, action, reward, None, state2, done)
            
            replay.append(exp)
            state1 = state2
            rewards.append(reward)
            
            # print out
            print(action)
            print(reward)
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                Q1, Q2, X, Y, loss = DQNModel.batch_update(minibatch, target_net, DQNModel.state_dim)

                print(i, loss.item())
                clear_output(wait=True)
                
                DQNModel.optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                DQNModel.optimizer.step()
                
                if j % sync_freq == 0:
                    target_net.load_state_dict(DQNModel.model.state_dict())

            if done or mov > max_steps:
                avg_episode_reward = np.mean(np.array(rewards))
                clear_output(wait=True)
                episode_rewards.append(avg_episode_reward)
                status = 0
                mov = 0
                
        average_episode_rewards.append(np.mean(np.array(episode_rewards)[-50:] ))
    
    return np.array(losses), np.array(episode_rewards), np.array(average_episode_rewards)

def run_dqn_eval(DQNModel, 
            marketEnv = MarketEnv(action_size = ACTION_DIM), 
            episodes = episodes, 
            max_steps = MAX_STEPS):
    
    episode_rewards = []
    average_episode_rewards = []

    for i in range(episodes):
        state1_ = marketEnv.reset()
        state1 = torch.from_numpy(state1_).float().to(device = devid)
        
        status = 1
        mov = 0
        rewards = []
        
        while(status == 1): 
            mov += 1
            qval = DQNModel(state1)
            
            if not torch.cuda.is_available():
                qval_ = qval.data.numpy()
            else:
                qval_ = qval.data.cpu().numpy()
            
            action_ind = np.argmax(qval_)
            
            # Execute action and upate state, and get reward + boolTerminal
            action = action_ind
            marketEnv.step(action)
            state2_, reward, done, info_dic = marketEnv.step(action)
            state2 = torch.from_numpy(state2_).float().to(device = devid)
            exp = (state1, action, reward, None, state2, done)
            
            replay.append(exp)
            state1 = state2
            rewards.append(reward)

            # print out
            print(action)
            print(reward)
            
            if done or mov > max_steps:
                
                avg_episode_reward = np.mean(np.array(rewards))
                clear_output(wait=True)
                episode_rewards.append(avg_episode_reward)
                status = 0
                mov = 0
                
        average_episode_rewards.append(np.mean(np.array(episode_rewards)[-50:] ))
    
    return np.array(episode_rewards), np.array(average_episode_rewards)

if __name__ == '__main__':
    run_dqn(DQNModel)