import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import torch
from torch.utils import tensorboard as tb
from tqdm import tqdm

from models.ddpg_attacker import FixedBudgetNetworkedWideDDPG
from models.dl.allocators import DDPGAllocator
from models.dl.component import DDPGComponent
from models.dl.epsilon import ConstantEpsilon
from models.dl.noise import OUActionNoise, GaussianNoiseDecay
from models.heuristics.allocators import ProportionalAllocator
from models.heuristics.budgeting import FixedBudgeting
from models.heuristics.component import GreedyComponent
from models.rl_attackers import Attacker
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment
from util.rl.experience_replay import ExperienceReplay


def exception_wrapper(config):
    try:
        return train_single(config)
    except:
        return 0.0


def train_single(config):
    time.sleep(config['seed'])
    model_name = config['model_name']

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    base_path = '.'

    # run_id = f'{datetime.now().strftime("%Y%m%d%H%M%S%f")}'
    run_id = f'{datetime.now().strftime("%Y%m%d%H%M%S%f")}-{config["budget"]}-{config["city"]}-{config["model_name"]}'
    # run_id = f'{datetime.now().strftime("%Y%m%d%H%M%S%f")},erf={env_randomize_factor:.5f},a_lr={allocator_actor_lr:.5f},c_lr={allocator_critic_lr:.5f},a_g={allocator_gamma:.5f},a_l={allocator_lam:.5f},a_e={allocator_epsilon:.5f},a_ec={allocator_entropy_coeff:.5f},a_vc={allocator_value_coeff:.5f},a_nu={allocator_n_updates},a_pgc={allocator_policy_grad_clip},a_bs={allocator_batch_size},a_crv={allocator_clip_range_vf},ge={greedy_epsilon:.5f}'
    # run_id = f'{datetime.now().strftime("%Y%m%d%H%M%S%f")},c_lr={critic_lr:.5f},a_lr={actor_lr:.5f},t={tau:.5f},dl={decay_length},ut={update_time}'
    writer = tb.SummaryWriter(f'{base_path}/logs/{run_id}')
    os.makedirs(f'{base_path}/logs/{run_id}/weights')

    log_handlers = [
        logging.FileHandler(f'{base_path}/logs/{run_id}/log.log'),
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
            method='network_file',
            city=config['city'],
        ),
        horizon=config['horizon'],
        num_sample=20,
        render_mode=None,
        congestion=True,
        trips=dict(
            type='trips_file',
            # type='trips_file_demand',
            randomize_factor=config['randomize_factor'],
        ),
        # rewarding_rule='normalized',
        rewarding_rule='proportional',
        reward_multiplier=1.0,
        n_components=config['n_components'],
    )

    env = DynamicMultiAgentTransportationNetworkEnvironment(env_config, base_path=base_path)

    total_steps = config['n_steps'] * config['n_epochs']

    if model_name == 'FixedBudgetNetworkedWideDDPG':
        model = FixedBudgetNetworkedWideDDPG(
            edge_component_mapping=env.edge_component_mapping,
            budgeting=FixedBudgeting(
                budget=config['budget'],
                noise=OUActionNoise(0, 0.5, 0.001, 30_000),
            ),
            n_features=5,
            actor_lr=config['actor_lr'],
            critic_lr=config['critic_lr'],
            gamma=config['gamma'],
            tau=config['tau'],
            noise=GaussianNoiseDecay(0.2, 0.00001, config['decay_length'])
        )
    elif model_name == 'FixedBudgetDDPGAllocatorDDPGComponent':
        model = Attacker(
            edge_component_mapping=env.edge_component_mapping,
            budgeting=FixedBudgeting(
                budget=config['budget'],
                noise=OUActionNoise(0, 0.5, 0.001, 30_000),
            ),
            name='FixedBudgetDDPGAllocatorDDPGComponent',
            allocator=DDPGAllocator(
                env.edge_component_mapping,
                2,
                critic_lr=config['allocator/critic_lr'],
                actor_lr=config['allocator/actor_lr'],
                gamma=config['allocator/gamma'],
                tau=config['allocator/tau'],
                noise=GaussianNoiseDecay(0.5, 0.001, config['allocator/decay_length']),
                epsilon=ConstantEpsilon(0.0)
            ),
            component=DDPGComponent(
                env.edge_component_mapping,
                config['component/n_features'],
                critic_lr=config['component/critic_lr'],
                actor_lr=config['component/actor_lr'],
                gamma=config['component/gamma'],
                tau=config['component/tau'],
                noise=GaussianNoiseDecay(0.2, 0.00001, config['component/decay_length']),
            )
        )
    elif model_name == 'FixedBudgetProportionalAllocatorDDPGComponent':
        model = Attacker(
            edge_component_mapping=env.edge_component_mapping,
            budgeting=FixedBudgeting(
                budget=config['budget'],
                noise=OUActionNoise(0, 0.5, 0.001, 30_000),
            ),
            name='FixedBudgetProportionalAllocatorDDPGComponent',
            allocator=ProportionalAllocator(),
            component=DDPGComponent(
                env.edge_component_mapping,
                config['component/n_features'],
                critic_lr=config['component/critic_lr'],
                actor_lr=config['component/actor_lr'],
                gamma=config['component/gamma'],
                tau=config['component/tau'],
                noise=GaussianNoiseDecay(0.2, 0.00001, config['component/decay_length']),
            )
        )
    elif model_name == 'FixedBudgetDDPGAllocatorGreedyComponent':
        model = Attacker(
            edge_component_mapping=env.edge_component_mapping,
            budgeting=FixedBudgeting(
                budget=config['budget'],
                noise=OUActionNoise(0, 0.5, 0.001, 30_000),
            ),
            name='FixedBudgetDDPGAllocatorGreedyComponent',
            allocator=DDPGAllocator(
                env.edge_component_mapping,
                2,
                critic_lr=config['allocator/critic_lr'],
                actor_lr=config['allocator/actor_lr'],
                gamma=config['allocator/gamma'],
                tau=config['allocator/tau'],
                noise=GaussianNoiseDecay(0.5, 0.001, config['allocator/decay_length']),
                epsilon=ConstantEpsilon(0.0)
            ),
            component=GreedyComponent(env.edge_component_mapping),
        )
    else:
        raise Exception(f'Unknown model {model_name}')

    logger.info(model)

    with open(f'{base_path}/logs/{run_id}/config.json', 'w') as fd:
        json.dump(env_config, fd, indent=4)

    device = torch.device('cpu')
    logger.info(f'Computation device: {device}')
    logger.info(env_config)

    buffer = ExperienceReplay(100_000, 64)
    # buffer = TrajectoryExperience()
    model = model.to(device)
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

    while global_step < total_steps:

        for _ in range(config['n_steps']):

            pbar.update(1)
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

            if global_step % config['update_time'] == 0:

                stats = model.update(*buffer.get_experiences())

                for name, value in stats.items():
                    if type(value) is list:
                        writer.add_histogram(name, np.array(value), global_step)
                    elif type(value) is np.ndarray:
                        writer.add_histogram(name, value, global_step)
                    else:
                        writer.add_scalar(name, value, global_step)

            # buffer.reset()

            if global_step % 1000 == 0:
                torch.save(model, f'{base_path}/logs/{run_id}/weights/Attacker_{global_step}.tar')

        # yield dict(episode_reward=np.mean(original_reward_history[-10:]))

    # writer.add_hparams(config, metric_dict={'reward': np.mean(original_reward_history[-10:])})
    torch.save(model, f'{base_path}/logs/{run_id}/weights/Attacker_final.tar')

    return run_id, np.mean(original_reward_history[-10:])


if __name__ == '__main__':

    # train_single(
    #     {
    #         'seed': 0,
    #         'log_stdout': True,
    #         'model_name': 'FixedBudgetNetworkedWideDDPG',
    #         'city': 'SiouxFalls',
    #         'horizon': 400,
    #         'randomize_factor': 0.01,
    #         'budget': 30,
    #         'n_components': 4,
    #         'n_steps': 1024,
    #         'n_epochs': 1,
    #         'update_time': 1,
    #         'critic_lr': 0.001,
    #         'actor_lr': 0.00001,
    #         'gamma': 0.99,
    #         'tau': 0.001,
    #         'decay_length': 30_000
    #     }
    # )
    #
    # train_single(
    #     {
    #         'seed': 0,
    #         'log_stdout': True,
    #         'city': 'EMA',
    #         'horizon': 400,
    #         'model_name': 'FixedBudgetDDPGAllocatorDDPGComponent',
    #         'randomize_factor': 0.01,
    #         'budget': 30,
    #         'n_components': 4,
    #         'n_steps': 1024,
    #         'n_epochs': 1,
    #         'update_time': 1,
    #         'allocator/n_features': 2,
    #         'allocator/critic_lr': 0.001,
    #         'allocator/actor_lr': 0.00001,
    #         'allocator/gamma': 0.99,
    #         'allocator/tau': 0.001,
    #         'allocator/decay_length': 30_000,
    #         'component/n_features': 5,
    #         'component/critic_lr': 0.001,
    #         'component/actor_lr': 0.00001,
    #         'component/gamma': 0.99,
    #         'component/tau': 0.001,
    #         'component/decay_length': 10_000,
    #     }
    # )
    #
    # train_single(
    #     {
    #         'seed': 0,
    #         'log_stdout': True,
    #         'city': 'EMA',
    #         'horizon': 400,
    #         'model_name': 'FixedBudgetProportionalAllocatorDDPGComponent',
    #         'randomize_factor': 0.01,
    #         'budget': 30,
    #         'n_components': 4,
    #         'n_steps': 1024,
    #         'n_epochs': 1,
    #         'update_time': 1,
    #         'component/n_features': 5,
    #         'component/critic_lr': 0.001,
    #         'component/actor_lr': 0.005,
    #         'component/gamma': 0.99,
    #         'component/tau': 0.001,
    #         'component/decay_length': 10_000,
    #     }
    # )
    #
    # train_single(
    #     {
    #         'seed': 0,
    #         'model_name': 'FixedBudgetDDPGAllocatorGreedyComponent',
    #         'log_stdout': True,
    #         'city': 'EMA',
    #         'horizon': 400,
    #         'randomize_factor': 0.01,
    #         'budget': 30,
    #         'n_components': 4,
    #         'n_steps': 1024,
    #         'n_epochs': 1,
    #         'update_time': 1,
    #         'allocator/n_features': 2,
    #         'allocator/critic_lr': 0.01,
    #         'allocator/actor_lr': 0.0001,
    #         'allocator/gamma': 0.99,
    #         'allocator/tau': 0.001,
    #         'allocator/decay_length': 10_000,
    #     }
    # )

    parameters = []

    # High-Level Runs

    for critic_lr in [0.01, 0.001, 0.005]:
        for actor_lr in [0.001, 0.001, 0.005, 0.0005, 0.00005]:
            for decay_length in [10_000, 30_000]:
                for n_components in [4, 6, 8, 10]:
                    for budget in [5, 15]:
                        for seed in range(2):
                            parameters.append({
                                'seed': seed,
                                'model_name': 'FixedBudgetDDPGAllocatorGreedyComponent',
                                'log_stdout': False,
                                'city': 'EMA',
                                'horizon': 400,
                                'randomize_factor': 0.01,
                                'budget': budget,
                                'n_components': n_components,
                                'n_steps': 1024,
                                'n_epochs': 100,
                                'update_time': 1,
                                'allocator/n_features': 2,
                                'allocator/critic_lr': critic_lr,
                                'allocator/actor_lr': actor_lr,
                                'allocator/gamma': 0.99,
                                'allocator/tau': 0.001,
                                'allocator/decay_length': decay_length,
                            })

    # Low-Level Runs

    for critic_lr in [0.01]:
        for actor_lr in [0.00005]:
            for decay_length in [10_000]:
                for n_components in [4, 6, 8, 10]:
                    for budget in [5, 10, 15, 30]:
                        for seed in range(2):
                            parameters.append({
                                'seed': seed,
                                'log_stdout': False,
                                'city': 'EMA',
                                'horizon': 400,
                                'model_name': 'FixedBudgetProportionalAllocatorDDPGComponent',
                                'randomize_factor': 0.01,
                                'budget': budget,
                                'n_components': n_components,
                                'n_steps': 1024,
                                'n_epochs': 200,
                                'update_time': 1,
                                'component/n_features': 5,
                                'component/critic_lr': critic_lr,
                                'component/actor_lr': actor_lr,
                                'component/gamma': 0.99,
                                'component/tau': 0.001,
                                'component/decay_length': decay_length,
                            })

    # Run HMADDPG

    for a_critic_lr in [0.01, 0.005, 0.0005]:
        for a_actor_lr in [0.0001, 0.00001, 0.00005]:
            for a_decay_length in [10_000, 50_000]:
                for a_gamma in [0.99, 0.9]:
                    for budget in [5, 10, 15, 30]:
                        for c_critic_lr in [0.01, 0.001]:
                            for c_actor_lr in [0.001, 0.0001]:
                                for c_gamma in [0.9, 0.99]:
                                    for n_components in [4, 6, 8, 10]:
                                        for c_decay_length in [10_000, 50_000]:
                                            for seed in range(2):
                                                parameters.append({
                                                    'seed': seed,
                                                    'log_stdout': False,
                                                    'city': 'EMA',
                                                    'horizon': 400,
                                                    'model_name': 'FixedBudgetDDPGAllocatorDDPGComponent',
                                                    'randomize_factor': 0.01,
                                                    'budget': budget,
                                                    'n_components': n_components,
                                                    'n_steps': 1024,
                                                    'n_epochs': 300,
                                                    'update_time': 1,
                                                    'allocator/n_features': 2,
                                                    'allocator/critic_lr': a_critic_lr,
                                                    'allocator/actor_lr': a_actor_lr,
                                                    'allocator/gamma': a_gamma,
                                                    'allocator/tau': 0.001,
                                                    'allocator/decay_length': a_decay_length,
                                                    'component/n_features': 5,
                                                    'component/critic_lr': c_critic_lr,
                                                    'component/actor_lr': c_actor_lr,
                                                    'component/gamma': c_gamma,
                                                    'component/tau': 0.001,
                                                    'component/decay_length': c_decay_length,
                                                })

    # RUN SingleDDPG

    for critic_lr in [0.01, 0.001]:
        for actor_lr in [0.0005, 0.00005]:
            for gamma in [0.99, 0.9]:
                for budget in [5, 10, 15, 30]:
                    for seed in range(2):
                        parameters.append({
                            'seed': seed,
                            'log_stdout': False,
                            'model_name': 'FixedBudgetNetworkedWideDDPG',
                            'city': 'SiouxFalls',
                            'horizon': 400,
                            'randomize_factor': 0.01,
                            'budget': budget,
                            'n_components': 4,
                            'n_steps': 1024,
                            'n_epochs': 100,
                            'update_time': 1,
                            'critic_lr': critic_lr,
                            'actor_lr': actor_lr,
                            'gamma': gamma,
                            'tau': 0.001,
                            'decay_length': 30_000
                        })

    # Running

    print(f'Running {len(parameters)} experiments')

    writer = tb.SummaryWriter('logs/hparam_search')

    with Pool(8) as pool:
        for param, (run_id, reward) in zip(parameters, tqdm(pool.imap(exception_wrapper, parameters), total=len(parameters))):
            writer.add_hparams(param, metric_dict={'reward': reward},
                               run_name=run_id)
