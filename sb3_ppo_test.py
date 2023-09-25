from stable_baselines3 import PPO
import gym

env = gym.make('CartPole-v1')

model = PPO(
    'MlpPolicy',
    env,
)

model.learn(total_timesteps=10000)