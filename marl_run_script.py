from re import S
import numpy as np
from dqn.marl_functions import *
from dqn.marl_agent import *

env = MarketEnv(action_size = 10, 
                    n_agents = 3, 
                    max_inventory = 0, 
                    max_demand = 3, 
                    demand_slope = 0.75)

marl_agent = MARLAgent(env)

episode_rewards, epoch_rewards, global_rewards, agent_rewards, losses, losses_eps, losses_nash  = run_marl(marl_agent, 
                                                                                marketEnv = env,
                                                                                batch_size = BATCH_SIZE,
                                                                                epochs = EPOCHS,
                                                                                explore_epsilon = EXPLORE_EPSILON,
                                                                                max_steps = MAX_STEPS,
                                                                                sync_freq = SYNC_FREQ,
                                                                                agent_index = 0)

env_id = "market-marl-nash"
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

