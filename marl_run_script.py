from re import S
import numpy as np
from marl_functions import *
from marl_agent import *

# BATCH_SIZE = 200

env = MarketEnv(action_size = 4, 
                    n_agents = 2, 
                    max_inventory = 3, 
                    max_demand = 3, 
                    demand_slope = 0.75)

marl_agent = MARLAgent(env)

losses, episode_rewards, epoch_rewards, global_rewards, agent_rewards = run_marl(marl_agent, 
                                                                                marketEnv = env,
                                                                                batch_size = BATCH_SIZE,
                                                                                epochs = 199,
                                                                                explore_epsilon = 0.2,
                                                                                max_steps = 10,
                                                                                sync_freq = 10,
                                                                                agent_index = 0)

env_id = "market-marl-40"
np.savetxt("./output/%s_dqn_losses.txt"%env_id, losses)
np.savetxt("./output/%s_dqn_epoch_rewards.txt"%env_id, epoch_rewards)

plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("Episodes",fontsize=22)
plt.ylabel("Loss",fontsize=22)
plt.savefig("./output/%s_dqn_losses.png"%env_id)

plt.figure(figsize=(10,7))
plt.plot(epoch_rewards)
plt.xlabel("Epochs",fontsize=22)
plt.ylabel("Avg Reward",fontsize=22)
plt.savefig("./output/%s_dqn_avg_reward.png"%env_id)

# episode_rewards_eval, epoch_rewards_eval = run_dqn_eval(DQNModel, 
#                                               marketEnv = MarketEnv(action_size = ACTION_DIM), 
#                                               epochs = 4000, 
#                                               max_steps = MAX_STEPS)

# plt.figure(figsize=(10,7))
# plt.plot(epoch_rewards_eval)
# plt.xlabel("Epochs",fontsize=22)
# plt.ylabel("Avg Reward",fontsize=22)

# perhaps pickle and save the model
torch.save(marl_agent, "./output/%s_dqn_model"%env_id)
# dqn2 = torch.load("./output/%s_dqn_model.png"%env_id)

# plt.figure(figsize=(10,7))
# plt.plot(epoch_rewards)
# plt.plot(epoch_rewards_eval)
# plt.xlabel("Epochs",fontsize=22)
# plt.ylabel("Avg Reward",fontsize=22)
# plt.savefig("./output/%s_dqn_avg_reward.png"%env_id)

np.mean(epoch_rewards)

print("done")

