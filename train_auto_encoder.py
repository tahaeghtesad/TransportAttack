import random

import numpy as np
import torch
import tqdm
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset

from models.autoencoders.action_ae import ConditionalAutoEncoder
from transport_env.GymMultiEnv import ComponentTrainer

pbar = tqdm.tqdm()


def collect_data(index, n_samples):

    try:
        return np.load(f'saved_partial_models/dataset_observation_{index}.npy'), np.load(f'saved_partial_models/dataset_action_{index}.npy')
    except FileNotFoundError:
        pass

    pbar.set_description(f'Collecting data {index}')
    pbar.set_postfix({})
    env_config = dict(
        network=dict(
            method='network_file',
            city='SiouxFalls',
            randomize_factor=0.00,
        ),
        horizon=50,
        render_mode=None,
        congestion=True,
        rewarding_rule='proportional',
        reward_multiplier=1.0,
        n_components=10,
    )
    edge_component_mapping = [
        [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 18, 22, 30, 34],
        [12, 15, 16, 17, 19, 20, 21, 23, 24, 25, 28, 31, 42, 46, 47, 49, 50, 51, 53, 54, 59],
        [6, 9, 26, 32, 33, 35, 36, 37, 38, 39, 41, 43, 65, 69, 70, 72, 73, 75],
        [27, 29, 40, 44, 45, 48, 52, 56, 57, 58, 55, 60, 61, 62, 63, 64, 66, 67, 68, 71, 74]
    ]

    def create(horizon):
        def env_creator():
            config = env_config.copy()
            config['horizon'] = horizon
            return ComponentTrainer(config, edge_component_mapping, index)

        return env_creator

    model = PPO.load(f'saved_partial_models/ppo_{index}_final.zip')

    eval_env = create(50)()
    action_list = []
    observation_list = []

    done = False
    obs, _ = eval_env.reset()
    pbar.reset(n_samples)
    observation_list.append(obs)
    for step in range(n_samples):
        pbar.update(1)
        action, _ = model.predict(obs, deterministic=random.random() < 0.5)
        action_list.append(action)
        obs, _, done, _, _ = eval_env.step(action)
        if done:
            obs, _ = eval_env.reset()
        observation_list.append(obs)

    action_list = np.array(action_list)
    observation_list = np.array(observation_list[:-1])
    np.save(f'saved_partial_models/dataset_observation_{index}.npy', observation_list)
    np.save(f'saved_partial_models/dataset_action_{index}.npy', action_list)
    return observation_list, action_list


def train_autoencoder(observations, actions, index, n_epochs):
    pbar.set_description('Training autoencoder')
    device = 'cuda:1'
    model = ConditionalAutoEncoder(observations.shape[1], actions.shape[1], 10, 2).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    data_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.array(observations, dtype=np.float32)).to(device),
            torch.from_numpy(np.array(actions, dtype=np.float32)).to(device)
        ), batch_size=64, shuffle=True
    )
    pbar.reset(n_epochs)
    for _ in range(n_epochs):
        pbar.update(1)
        losses = []
        for observations, actions in data_loader:
            y_action = model.forward(observations, actions)
            loss = criterion(y_action, torch.nn.functional.normalize(actions, dim=1, p=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(dict(loss=losses[-1], mean_loss=np.mean(losses)))

    torch.save(model, f'saved_partial_models/decoder_{index}.pt')


if __name__ == '__main__':
    for i in range(4):
        observation, action = collect_data(i, 50_000)
        train_autoencoder(observation, action, i, 100)
