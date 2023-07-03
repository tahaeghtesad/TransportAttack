from ray.rllib.algorithms.dqn import DQNConfig
from tqdm import tqdm

algo = (
    DQNConfig()
    .environment("CartPole-v1")
    .rollouts(num_rollout_workers=1, rollout_fragment_length=512)
    .framework("torch")
    .training(model=dict(fcnet_hiddens=[256, 256]), gamma=0.99)
    .evaluation(evaluation_num_workers=1)
    .resources(num_gpus=1)
).build()

for _ in (pbar := tqdm(range(100))):
    res = algo.train()
    pbar.set_description(f'Episode reward: {res["episode_reward_mean"]}, Episode Max Reward: {res["episode_reward_max"]}, Episode Min Reward: {res["episode_reward_min"]}, Episode Length: {res["episode_len_mean"]}')