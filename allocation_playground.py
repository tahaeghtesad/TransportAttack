import logging
import sys
from datetime import datetime
from typing import Optional, Union, Tuple

import gym
import numpy as np
import torch
from torch.utils import tensorboard as tb
from gym.core import ObsType, ActType
from tqdm import tqdm

from models import CustomModule
from util.rl.experience_replay import TrajectoryExperience, BasicTrajectoryExperience
from util.torch.rl import GeneralizedAdvantageEstimation


class AllocationPlayGroundStateless(gym.Env):

    def __init__(self, ndim=5):
        super().__init__()
        self.ndim = ndim
        self.target = np.eye(self.ndim, dtype=np.float32)
        self.values = np.random.randint(low=0, high=10, size=self.ndim)
        self.ntries = 0

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        assert len(action) == self.ndim
        assert np.abs(np.sum(action) - 1.0) < 1e-6, f'Action must sum to 1.0, got {np.sum(action)}'
        reward = -np.linalg.norm(self.target[self.ntries] - action) * 1000
        self.ntries += 1
        return np.array([self.ntries]), reward, False, {'TimeLimit.truncated': self.ntries < 50}
        # return np.array([np.random.randint(0, 100)]), reward, False, {'TimeLimit.truncated': self.ntries < 50}

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Union[
        ObsType, tuple[ObsType, dict]]:
        self.ntries = 0
        return np.zeros(1)

    def render(self, mode="human"):
        raise NotImplementedError()


class SimplePPO(CustomModule):

    def __init__(self, entropy=0.001, ndim=5):
        super().__init__('SimplePPO')
        self.ndim = ndim

        self.gamma = 0.0
        self.lam = 1.00
        self.lr = 0.001
        self.batch_size = 64
        self.n_epochs = 10
        self.epsilon = 0.3
        self.entropy = entropy

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, ndim),
            torch.nn.Softplus(),
        )
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        self.gae = GeneralizedAdvantageEstimation(gamma=self.gamma, lam=self.lam)

    def get_distribution(self, logits):
        return torch.distributions.Dirichlet(concentration=logits + 1)

    def forward(self, state):
        states = torch.unsqueeze(torch.tensor(state, dtype=torch.float32, device=self.device), dim=0)
        logits = self.actor(states)
        distribution = self.get_distribution(logits)
        action = distribution.sample()
        return action[0].detach().cpu().numpy()

    def update(self, states, actions, rewards, dones, truncateds):
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
        truncateds = torch.tensor(np.array(truncateds), dtype=torch.float32, device=self.device)

        n_samples = states.shape[0]
        assert n_samples % self.batch_size == 0, 'Batch size must be a multiple of n_samples'

        with torch.no_grad():
            old_values = self.critic.forward(states)
            advantages = self.gae.forward(old_values, rewards, dones, truncateds)
            returns = advantages + old_values
            old_logits = self.actor.forward(states)
            old_distribution = self.get_distribution(old_logits)
            old_log_prob = torch.unsqueeze(old_distribution.log_prob(actions), dim=1)

        for epoch in range(self.n_epochs):
            permutations = torch.randperm(n_samples)
            for batch in range(n_samples // self.batch_size):
                indices = permutations[batch * self.batch_size: (batch + 1) * self.batch_size]

                values = self.critic(states[indices])

                critic_loss = torch.nn.functional.mse_loss(values, returns[indices])
                batch_adv = (advantages[indices] - advantages[indices].mean()) / (advantages[indices].std() + 1e-8)
                logits = self.actor(states[indices])
                distribution = self.get_distribution(logits)
                new_log_prob = torch.unsqueeze(distribution.log_prob(actions[indices]), dim=1)
                log_ratio = new_log_prob - old_log_prob[indices]
                surr1 = log_ratio * batch_adv
                surr2 = torch.clamp(log_ratio, np.log(1 - self.epsilon), np.log(1 + self.epsilon)) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                entropy = - distribution.entropy().mean()
                loss = actor_loss + 1.0 * critic_loss - self.entropy * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


        with torch.no_grad():
            logits = self.actor.forward(states)
            distribution = self.get_distribution(logits)
            log_prob = torch.unsqueeze(distribution.log_prob(actions), dim=1)
            entropy = distribution.entropy()

        return {
            'actor_loss': actor_loss.detach().cpu().numpy().item(),
            'critic_loss': critic_loss.detach().cpu().numpy().item(),
            'advantages': advantages.detach().cpu().numpy().mean(),
            'max_concentration': old_logits.detach().cpu().numpy().max(),
            'min_concentration': old_logits.detach().cpu().numpy().min(),
            'max_entropy': entropy.detach().cpu().numpy().max(),
            'max_log_prob': log_prob.detach().cpu().numpy().max(),
            'min_log_prob': log_prob.detach().cpu().numpy().min(),
            'max_log_prob_change': torch.abs(log_prob - old_log_prob).detach().cpu().numpy().max(),
        }


def run(entropy):
    torch.set_num_threads(1)
    logging.basicConfig(
        format='[%(asctime)s] [%(name)s] [%(threadName)s] [%(levelname)s] - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    writer = tb.SummaryWriter(f'logs/{datetime.now().strftime("%Y%m%d%H%M%S%f")}')
    logger = logging.getLogger(__name__)

    # ndim = 85
    ndim = 4
    env = AllocationPlayGroundStateless(ndim=ndim)
    model = SimplePPO(ndim=ndim, entropy=entropy)
    replay_buffer = BasicTrajectoryExperience()

    n_steps = 2048
    total_steps = n_steps * 128

    device = torch.device('cpu')
    model.to(device)
    logger.info(f'Device: {device}')
    logger.info(f'{model}')
    pbar = tqdm(total=total_steps, desc='Training')
    global_step = 0
    total_samples = 0
    state = env.reset()
    done = False
    truncated = False
    rewards = 0
    step = 0
    reward = 0
    episode = 0

    while global_step < total_steps:

        for _ in range(n_steps):
            pbar.update(1)
            step += 1
            global_step += 1

            action = model.forward(state)
            next_state, reward, done, info = env.step(action)

            rewards += reward
            truncated = info['TimeLimit.truncated']
            replay_buffer.add(state, action, reward, done, truncated)
            state = next_state

            if done or truncated:
                state = env.reset()
                writer.add_scalar('episode/reward', -rewards, global_step)
                rewards = 0
                step = 0
                episode += 1

        stats = model.update(*replay_buffer.get_experiences())
        for name, value in stats.items():
            writer.add_scalar(f'train/{name}', value, global_step)

        writer.add_scalar('episode/entropy', entropy, global_step)
        replay_buffer.reset()


if __name__ == '__main__':
    for entropy in [0.001, 0.01, 0.1, 1.0, 10.0]:
        for _ in range(5):
            run(entropy)