import json
import logging
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from models.agents.heuristics.attackers.budgeting import FixedBudgeting
from models.agents.rl_agents.attackers.ddpg_attacker import FixedBudgetNetworkedWideTD3
from models.agents.rl_agents.attackers.ppo_attacker import FixedBudgetNetworkedWidePPO
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment
from util.rl.experience_replay import TrajectoryExperience, ExperienceReplay
from util.torch.writer import TBStatWriter


def train_single(config, model_creator, mode):
    assert mode == 'off_policy' or mode == 'on_policy', f'mode must be either off_policy or on_policy, but got "{mode}"'
    time.sleep(random.random() * 10)
    torch.set_num_threads(1)

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    base_path = '.'
    log_name = 'logs'

    run_id = f'{datetime.now().strftime("%Y%m%d%H%M%S%f")}'
    # run_id = f'{datetime.now().strftime("%Y%m%d%H%M%S%f")}-{config["budget"]}-{config["city"]}-{config["model_name"]}'
    # run_id = f'{datetime.now().strftime("%Y%m%d%H%M%S%f")},erf={env_randomize_factor:.5f},a_lr={allocator_actor_lr:.5f},c_lr={allocator_critic_lr:.5f},a_g={allocator_gamma:.5f},a_l={allocator_lam:.5f},a_e={allocator_epsilon:.5f},a_ec={allocator_entropy_coeff:.5f},a_vc={allocator_value_coeff:.5f},a_nu={allocator_n_updates},a_pgc={allocator_policy_grad_clip},a_bs={allocator_batch_size},a_crv={allocator_clip_range_vf},ge={greedy_epsilon:.5f}'
    # run_id = f'{datetime.now().strftime("%Y%m%d%H%M%S%f")},c_lr={critic_lr:.5f},a_lr={actor_lr:.5f},t={tau:.5f},dl={decay_length},ut={update_time}'
    writer = TBStatWriter(f'{base_path}/{log_name}/{run_id}')
    os.makedirs(f'{base_path}/{log_name}/{run_id}/weights')

    log_handlers = [
        logging.FileHandler(f'{base_path}/{log_name}/{run_id}/log.log'),
    ]
    if config['log_stdout']:
        log_handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        format='[%(asctime)s] [%(name)s] [%(threadName)s] [%(levelname)s] - %(message)s',
        level=logging.INFO,
        handlers=log_handlers
    )

    logger = logging.getLogger('main')

    env_config = dict(
        network=dict(
            # method='network_file',
            # city='SiouxFalls',
            method='edge_list',
            file=config['file'],
            randomize_factor=config['randomize_factor']
        ),
        horizon=config['horizon'],
        render_mode=None,
        congestion=True,
        # rewarding_rule='normalized',
        # rewarding_rule='proportional',
        # rewarding_rule='travel_time_increased',
        rewarding_rule=config['rewarding_rule'],
        reward_multiplier=1.0,
        n_components=config['n_components'],
    )

    env = DynamicMultiAgentTransportationNetworkEnvironment(env_config, base_path=base_path)

    total_steps = config['n_steps'] * config['n_epochs']

    model = model_creator(env)

    logger.info(model)

    with open(f'{base_path}/{log_name}/{run_id}/config.json', 'w') as fd:
        json.dump(env_config, fd, indent=4)

    device = torch.device('cpu')
    logger.info(f'Computation device: {device}')
    logger.info(env_config)

    model = model.to(device)

    buffer = ExperienceReplay(100_000, 64) if mode == 'off_policy' else TrajectoryExperience()

    pbar = tqdm(total=total_steps, desc='Training', disable=not config['log_stdout'])

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
    original_reward_history = []
    best_episode_reward = 0

    while global_step < total_steps:

        for s_count in range(config['n_steps']):

            pbar.update(1)
            pbar.set_description(f'Collecting Sample {s_count}/{config["n_steps"]}')
            step += 1
            global_step += 1

            constructed_action, action, allocation, budget = model.forward_single(state, deterministic=False)
            next_state, reward, done, info = env.step(constructed_action)

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

                original_reward_history.append(original_rewards)

                if original_rewards > best_episode_reward:
                    best_episode_reward = original_rewards
                    torch.save(model, f'{base_path}/{log_name}/{run_id}/weights/Attacker_{global_step}.tar')

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

            if mode == 'off_policy':
                stats = model.update(*buffer.get_experiences())


                buffer.reset()

        if mode == 'on_policy':
            stats = model.update(*buffer.get_experiences())
            writer.add_stats(stats, global_step)

            buffer.reset()

    torch.save(model, f'{base_path}/{log_name}/{run_id}/weights/Attacker_final.tar')

    return run_id, best_episode_reward


if __name__ == '__main__':
    config = dict(
            # file='GRE-4x2-0.5051-0.1111-20231121124113244125',
            file='GRE-4x4-0.5051-0.1111-20231121131109967053',
            # file='GRE-6x8-0.5051-0.1111-20231121124104988621',
            seed=np.random.randint(1000),
            rewarding_rule='mixed',
            # rewarding_rule='travel_time_increased',
            horizon=50,
            randomize_factor=0.01,
            n_components=4,
            n_steps=2048,
            n_epochs=128,
            log_stdout=True,
        )

    def ppo_model_creator(env):
        return FixedBudgetNetworkedWidePPO(
            edge_component_mapping=env.edge_component_mapping,
            budgeting=FixedBudgeting(30),
            n_features=5,
            actor_lr=0.0003,
            critic_lr=0.0003,
            gamma=0.97,
            lam=0.95,
            epsilon=0.2,
            normalize_advantages=False,
            entropy_coeff=1.0,
            n_updates=1,
            batch_size=128,
            value_coeff=0.1,
        )

    def td3_model_creator(env):
        return FixedBudgetNetworkedWideTD3(
            env.edge_component_mapping,
            budgeting=FixedBudgeting(30),
            n_features=5,
            actor_lr=0.0003,
            critic_lr=0.0003,
            gamma=0.97,
            tau=0.001,
            actor_update_interval=3,
        )

    # train_single(config, ddpg_model_creator, 'off_policy')
    train_single(config, ppo_model_creator, 'on_policy')
    # train_single(config, td3_model_creator, 'off_policy')
    # config['rewarding_rule'] = 'travel_time_increased'
    # train_single(config, ddpg_model_creator, 'off_policy')
    # config['file'] = 'GRE-4x4-0.5051-0.1111-20231121131109967053'
    # train_single(config, ddpg_model_creator, 'off_policy')
    # train_single(config, ppo_model_creator, 'on_policy')

    # train_single(config, ppo_model_and_buffer_creator)
