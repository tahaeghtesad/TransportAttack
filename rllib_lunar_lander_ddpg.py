from ray.rllib.algorithms.ddpg import DDPGConfig

algo = (
    DDPGConfig()
    .environment("LunarLander-v2", env_config=dict(continuous=True))
    .rollouts(num_rollout_workers=6)
    .framework("tf2")
    .training(model=dict(fcnet_hiddens=[256, 256]), gamma=0.99)
    .evaluation(evaluation_num_workers=1)
).build()

for _ in range(100):
    res = algo.train()
    print(res)
    print(f'Episode reward: {res["episode_reward_mean"]}, Episode Max Reward: {res["episode_reward_max"]}, Episode Min Reward: {res["episode_reward_min"]}, Episode Length: {res["episode_len_mean"]}')

algo.evaluate()