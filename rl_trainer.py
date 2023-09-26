import json
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils import tensorboard as tb
from tqdm import tqdm

from models.dl.allocators import PPOAllocator
from models.dl.noise import OUActionNoise
from models.heuristics.budgeting import FixedBudgeting
from models.heuristics.component import GreedyComponent
from models.rl_attackers import Attacker
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment
from util.rl.experience_replay import TrajectoryExperience


def train_single(
        env_randomize_factor,
        allocator_actor_lr,
        allocator_critic_lr,
        allocator_gamma,
        allocator_lam,
        allocator_epsilon,
        allocator_entropy_coeff,
        allocator_value_coeff,
        allocator_n_updates,
        allocator_policy_grad_clip,
        allocator_batch_size,
        allocator_max_concentration,
        allocator_clip_range_vf,
        log_stdout=True,
):

    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)

    # run_id = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    run_id = f'erf={env_randomize_factor},a_lr={allocator_actor_lr},c_lr={allocator_critic_lr},a_g={allocator_gamma},a_l={allocator_lam},a_e={allocator_epsilon},a_ec={allocator_entropy_coeff},a_vc={allocator_value_coeff},a_nu={allocator_n_updates},a_pgc={allocator_policy_grad_clip},a_bs={allocator_batch_size},a_mc={allocator_max_concentration},a_crv={allocator_clip_range_vf},{datetime.now().strftime("%H%M%S")}'
    writer = tb.SummaryWriter(f'logs/{run_id}')
    os.makedirs(f'logs/{run_id}/weights')

    log_handlers = [
        logging.FileHandler(f'logs/{run_id}/log.log'),
    ]
    if log_stdout:
        log_handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        format='[%(asctime)s] [%(name)s] [%(threadName)s] [%(levelname)s] - %(message)s',
        level=logging.INFO,
        handlers=log_handlers
    )

    logger = logging.getLogger('main')

    env_config = dict(
        network=dict(
            method='network_file',
            city='SiouxFalls',
        ),
        horizon=50,
        num_sample=20,
        render_mode=None,
        congestion=True,
        trips=dict(
            type='trips_file',
            randomize_factor=env_randomize_factor,
        ),
        rewarding_rule='proportional',
        reward_multiplier=1.0,
        n_components=4,
    )

    env = DynamicMultiAgentTransportationNetworkEnvironment(env_config)

    n_steps = 512
    total_steps = n_steps * 192

    model = Attacker(
        name='FixedBudgetDDPGAllocatorGreedyComponent',
        edge_component_mapping=env.edge_component_mapping,
        budgeting=FixedBudgeting(
            budget=30,
            noise=OUActionNoise(0, 0.5, 0.01, 5_000),
        ),
        allocator=PPOAllocator(
            env.edge_component_mapping,
            5,
            # actor_lr=0.000005,
            actor_lr=allocator_actor_lr,
            critic_lr=allocator_critic_lr,
            gamma=allocator_gamma,
            lam=allocator_lam,
            epsilon=allocator_epsilon,
            value_coeff=allocator_value_coeff,
            entropy_coeff=allocator_entropy_coeff,
            n_updates=allocator_n_updates,
            # policy_grad_clip=0.05,
            # policy_grad_clip=None,
            policy_grad_clip=allocator_policy_grad_clip,
            batch_size=allocator_batch_size,
            max_concentration=allocator_max_concentration,
            # clip_range_vf=None
            clip_range_vf=allocator_clip_range_vf,
        ),
        component=GreedyComponent(
            env.edge_component_mapping,
        ),
    )

    logger.info(model)

    with open(f'logs/{run_id}/config.json', 'w') as fd:
        json.dump(env_config, fd, indent=4)

    device = torch.device('cpu')
    logger.info(f'Computation device: {device}')
    logger.info(env_config)

    # buffer = ExperienceReplay(100_000, 64)
    buffer = TrajectoryExperience()
    model = model.to(device)
    pbar = tqdm(total=total_steps, desc='Training')

    global_step = 0
    total_samples = 0

    state = env.reset()
    done = False
    truncated = False
    rewards = 0
    original_rewards = 0
    component_rewards = np.zeros(env.n_components)
    step = 0
    discounted_reward = 0
    original_reward = 0
    episode = 0

    while global_step < total_steps:

        for _ in range(n_steps):

            pbar.update(1)
            step += 1
            global_step += 1

            action, allocation, budget = model.forward_single(state, deterministic=random.random() < 0.5)
            next_state, reward, done, info = env.step(action)

            truncated = info.get("TimeLimit.truncated", False)
            original_reward = info['original_reward']
            buffer.add(state, allocation, budget, action, reward, next_state, done, truncated)
            state = next_state
            rewards += sum(reward)
            original_rewards += original_reward
            component_rewards += reward
            discounted_reward += sum(reward) * (0.97 ** step)
            total_samples += 1

            if done or truncated:

                writer.add_scalar(f'env/cumulative_reward', rewards, global_step)
                writer.add_scalar(f'env/discounted_reward', discounted_reward, global_step)
                writer.add_scalar(f'env/original_reward', original_rewards, global_step)
                writer.add_scalar(f'env/episode_length', step, global_step)

                for c in range(env.n_components):
                    writer.add_scalar(f'env/component_reward/{c}', component_rewards[c], global_step)

                pbar.set_description(
                    f'Episode {episode} | '
                    f'Len {step} | '
                    f'CumReward {rewards:.2f} | '
                    f'DisReward {discounted_reward:.3f} | '
                    f'ReplayBuffer {buffer.size()} | '
                    f'Global Step {global_step}'
                )

                state = env.reset()
                done = False
                truncated = False
                rewards = 0
                original_rewards = 0
                component_rewards = np.zeros(env.n_components)
                step = 0
                discounted_reward = 0
                original_reward = 0
                episode += 1

        try:
            stats = model.update(*buffer.get_experiences())
        except Exception as e:
            logger.error(e)
            return False

        for name, value in stats.items():
            if type(value) is list:
                writer.add_histogram(name, np.array(value), global_step)
            elif type(value) is np.ndarray:
                writer.add_histogram(name, value, global_step)
            else:
                writer.add_scalar(name, value, global_step)

        buffer.reset()

    return True
