from re import S
import numpy as np
from dqn.marl_functions import *
from dqn.marl_agent import *
from environment.MarketEnv import *
import pickle as pkl
from datetime import datetime
from common.properties import *
from dqn.marl_baseline import *

env = MarketEnv(action_size = ACTION_DIM, 
                    n_agents = N_AGENTS, 
                    max_inventory = 0, 
                    max_demand = MAX_DEMAND,
                    beta0 = BETA_0,
                    beta1 = BETA_1, 
                    beta2 = BETA_2, 
                    a = MARKET_A)

marl_agent = MARLAgent(env)
episodes_input = episodes

res = run_marl_baseline(marl_agent, 
                marketEnv = env,
                batch_size = BATCH_SIZE,
                episodes = episodes_input,
                explore_epsilon = EXPLORE_EPSILON,
                max_steps = MAX_STEPS,
                sync_freq = SYNC_FREQ,
                agent_index = 0)

now = datetime.now()
date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
env_id = "market-marl-maxq-" + date_time + str(episodes_input)
np.savetxt("./output/%s_dqn_losses.txt"%env_id, res.losses)
np.savetxt("./output/%s_dqn_episode_rewards.txt"%env_id, res.avg_episode_rewards)

plt.figure(figsize=(10,7))
plt.plot(res.losses)
plt.xlabel("Episodes", fontsize=22)
plt.ylabel("Loss", fontsize=22)
plt.savefig("./output/%s_dqn_losses.png"%env_id)

plt.figure(figsize=(10,7))
plt.plot(res.losses_eps)
plt.xlabel("Episodes", fontsize=22)
plt.ylabel("Loss", fontsize=22)
plt.savefig("./output/%s_dqn_nash_losses.png"%env_id)

plt.figure(figsize=(10,7))
plt.plot(res.losses_nash)
plt.xlabel("Episodes", fontsize=22)
plt.ylabel("Loss", fontsize=22)
plt.savefig("./output/%s_dqn_nash_losses.png"%env_id)

plt.figure(figsize=(10,7))
plt.plot(res.avg_episode_rewards)
# plt.plot(global_rewards)
plt.plot(res.avg_episode_rewards_agent)
plt.xlabel("Episodes",fontsize=22)
plt.ylabel("Avg Reward",fontsize=22)
plt.savefig("./output/%s_dqn_avg_reward.png"%env_id)

torch.save(marl_agent, "./output/%s_dqn_model"%env_id)
# dqn2 = torch.load("./output/%s_dqn_model.png"%env_id)

# plt.figure(figsize=(10,7))
# plt.xlabel("Episodes",fontsize=22)
# plt.ylabel("Avg Reward",fontsize=22)
# plt.savefig("./output/%s_dqn_avg_reward.png"%env_id)

result_filename = "./output/%s_results.pkl"%env_id
with open(result_filename, 'wb') as file:  # Overwrites any existing file.
    # pkl.dump(res, file, pkl.HIGHEST_PROTOCOL)
    pkl.dump(res, file)

print("done")

