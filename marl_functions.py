import numpy as np
import torch
from IPython.display import clear_output
import random
from collections import deque
from tests.test_gw import *
from environment.MarketEnv import MarketEnv
from common.properties import *
from dqn_net import DQNNet

def run_marl(MARLAgent, 
            marketEnv = MarketEnv(action_size = ACTION_DIM), 
            epochs = EPOCHS, 
            batch_size = BATCH_SIZE,
            max_steps = MAX_STEPS,
            sync_freq = SYNC_FREQ,
            explore_epsilon = EXPLORE_EPSILON):

    target_net = copy.deepcopy(MARLAgent.model)
    target_net.load_state_dict(MARLAgent.model.state_dict())

    episode_rewards = []
    avg_epoch_rewards = []
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
            for n in range(marketEnv.n_agents):
                play_prob = MARLAgent.prob_action(state1, 0)

                if (random.random() < explore_epsilon):
                    action_ind = np.random.randint(0, MARLAgent.output_size)
                else:
                    action_ind = np.argmax(play_prob)
                
                # Execute action and upate state, and get reward + boolTerminal
                action = action_ind
                marketEnv.step(action)
                state2_, reward, done, info_dic = marketEnv.step(action)
                state2 = torch.from_numpy(state2_).float().to(device = devid)
                exp = (state1, action, reward, state2, done)
                
                replay.append(exp)
                state1 = state2
                
                rewards.append(reward)

                # print out
                print(action)
                print(reward)
                
                if len(replay) > batch_size:
                    minibatch = random.sample(replay, batch_size)
                    Q1, Q2, X, Y, loss = MARLAgent.batch_update(minibatch, target_net, MARLAgent.state_dim)

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
                clear_output(wait=True)
                episode_rewards.append(avg_episode_reward)
                status = 0
                mov = 0
                
        avg_epoch_rewards.append(np.mean(np.array(episode_rewards)[-50:] ))
    
    return np.array(losses), np.array(episode_rewards), np.array(avg_epoch_rewards)