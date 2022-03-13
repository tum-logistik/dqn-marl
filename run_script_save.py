from dqn.dqn_functions import * 

STATE_DIM = 2
ACTION_DIM = 110
BATCH_SIZE = 200

DQNModel = DQNNet(state_dim = STATE_DIM, 
                  output_size = ACTION_DIM, 
                  hidden_size = 120,
                  batch_size = BATCH_SIZE,
                  n_agents = 1)

marketEnv1 = MarketEnv(action_size = ACTION_DIM)

losses, episode_rewards, epoch_rewards = run_dqn(DQNModel, 
                                                 marketEnv = marketEnv1,
                                                 batch_size = BATCH_SIZE,
                                                 epochs = 4001,
                                                 explore_epsilon = 0.2,
                                                 max_steps = 100,
                                                 sync_freq = 10)


env_id = "market-4001"
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
# plt.savefig("./output/%s_dqn_avg_reward.png"%env_id)

episode_rewards_eval, epoch_rewards_eval = run_dqn_eval(DQNModel, 
                                              marketEnv = MarketEnv(action_size = ACTION_DIM), 
                                              epochs = 4000, 
                                              max_steps = MAX_STEPS)

plt.figure(figsize=(10,7))
plt.plot(epoch_rewards_eval)
plt.xlabel("Epochs",fontsize=22)
plt.ylabel("Avg Reward",fontsize=22)

# perhaps pickle and save the model
torch.save(DQNModel, "./output/%s_dqn_model"%env_id)
# dqn2 = torch.load("./output/%s_dqn_model.png"%env_id)

plt.figure(figsize=(10,7))
plt.plot(epoch_rewards)
plt.plot(epoch_rewards_eval)
plt.xlabel("Epochs",fontsize=22)
plt.ylabel("Avg Reward",fontsize=22)
plt.savefig("./output/%s_dqn_avg_reward.png"%env_id)

np.mean(epoch_rewards_eval)


