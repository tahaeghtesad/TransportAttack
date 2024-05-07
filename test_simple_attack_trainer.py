import os
import random
from datetime import datetime

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from models.agents.heuristics.attackers.allocators import ProportionalAllocator
from models.agents.heuristics.attackers.budgeting import FixedBudgeting
from models.agents.heuristics.attackers.component import GreedyComponent
from models.agents.noisy_agent import NoisyAttacker, EpsilonGreedyAttacker
from models.agents.rl_agents.attackers.allocators.allocators import PPOAllocator
from models.agents.rl_agents.attackers.component.coppo_component import CoPPOComponent
from models.agents.rl_agents.attackers.component.mappo_component import MAPPOComponent
from models.agents.rl_agents.attackers.ddpg_attacker import FixedBudgetNetworkedWideTD3, FixedBudgetNetworkedWideDDPG
from models.agents.rl_agents.attackers.no_budget_td3_attacker import NoBudgetTD3Attacker
from models.agents.rl_agents.attackers.ppo_attacker import FixedBudgetNetworkedWidePPO
from models.agents.rl_agents.attackers.rl_attackers import Attacker
from models.exploration.epsilon import ConstantEpsilon, DecayEpsilon
from models.exploration.noise import GaussianNoiseDecay, ZeroNoise, GaussianNoise
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment
from util.math import sigmoid
from util.rl.experience_replay import ExperienceReplay, TrajectoryExperience
from util.scheduler import SimultaneousTrainingScheduler
from util.torch.writer import TBStatWriter


def __get_state_values_assuming_no_action(env, gamma, done):
    truncated = False
    immediate_rewards = []
    original_reward = []
    step_count = -1

    while not done and not truncated:
        step_count += 1
        action = np.zeros((env.base.number_of_edges(),))
        obs, reward, done, info = env.step(action)
        truncated = info.get('TimeLimit.truncated', False)
        original_reward.append(info.get('original_reward'))
        immediate_rewards.append(reward)

    immediate_rewards = np.array(immediate_rewards)
    state_value = np.zeros(immediate_rewards.shape[1])
    for i in range(immediate_rewards.shape[0]):
        state_value += gamma ** i * immediate_rewards[i]

    return state_value, np.sum(immediate_rewards, axis=0), step_count, original_reward


def train_simple_attacker(run_id, env, config, attacker_model):  # returns attacker strategy
    index = 'simple'

    writer = TBStatWriter(f'logs/{run_id}/attacker_{index}/')

    replay_buffer = ExperienceReplay(config['attacker_config']['buffer_size'],
                                     config['attacker_config']['batch_size'])

    done = False
    truncated = False
    obs = env.reset()
    episode_reward = []
    total_travel_time = []
    step_count = 0
    detected_at_step = env.config['horizon']
    episode = 0
    should_eval = episode % 10 == 0
    env_string = 'eval' if should_eval else 'env'

    for global_step in (pbar := tqdm(range(config['attacker_config']['training_steps']))):
        step_count += 1

        constructed_action, action, allocation, budget = attacker_model.forward_single(obs, deterministic=should_eval)
        # state_val = attacker_model.attacker.critic.forward(torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(constructed_action).unsqueeze(0))
        # writer.add_scalar(f'{env_string}/allocator_state_value', state_val.cpu().detach().numpy()[0], global_step)

        # bias = torch.sum(attacker_model.attacker.attacker.critic.adj * (torch.from_numpy(obs).unsqueeze(0)[:, :, [2]] + torch.from_numpy(obs).unsqueeze(0)[:, :, [3]]), dim=1)
        # writer.add_scalar(f'{env_string}/bias', bias.cpu().detach().numpy()[0], global_step)
        # writer.add_scalar(f'{env_string}/bias_value_diff', state_val.cpu().detach().numpy()[0] - bias.cpu().detach().numpy()[0], global_step)

        # perturbed_edge_travel_times = env.get_travel_times_assuming_the_attack(constructed_action)
        # detected = bool(np.random.random() < sigmoid(budget[0], sl=2.0, th=30.0))
        # detected = bool(budget[0] > 30)
        detected = False

        if detected:
            detected_at_step = step_count
            value, cumulative_reward, steps, original_reward = __get_state_values_assuming_no_action(env, config['attacker_config']['gamma'], done)
            # value, cumulative_reward, steps, original_reward = np.zeros((4,)), np.zeros((4,)), 0, np.zeros((4,))
            # value = np.zeros((4,))
            # value = np.zeros((1, ))
            replay_buffer.add(
                obs, allocation, budget, action, value, obs, True, False)
            episode_reward.append(cumulative_reward)
            total_travel_time.extend(original_reward)
            step_count += steps
            writer.add_scalar(f'{env_string}/detected_value', np.sum(value), global_step)
            done = True
        else:
            next_obs, reward, done, info = env.step(
                constructed_action
            )
            truncated = info.get('TimeLimit.truncated', False)
            total_travel_time.append(info.get('original_reward'))
            replay_buffer.add(obs, allocation, budget,
                              action, reward, next_obs,
                              done, truncated)
            obs = next_obs
            episode_reward.append(reward)
            writer.add_histogram(f'{env_string}/component_time_diff', info['component_time_diff'], global_step)

        writer.add_scalar(f'env/buffer_size', replay_buffer.size(), global_step)

        if replay_buffer.size() > \
                config['attacker_config']['batch_size']:
            observations, allocations, budgets, actions, rewards, next_observations, dones, truncateds = replay_buffer.get_experiences()
            stats = attacker_model.update(observations, allocations, budgets, actions, rewards,
                                          next_observations, dones, truncateds)
            writer.add_stats(stats, global_step)

        writer.add_scalar(f'{env_string}/attacker_budget', budget, global_step)
        writer.add_histogram(f'{env_string}/attacker_allocation', allocation, global_step)

        # TODO if this returns true, we should reset the experience replay buffer
        # attacker_model.attacker.iterative_scheduler.step()

        if done or truncated:
            pbar.set_description(
                f'Training Attacker |'
                f' ep: {episode} |'
                f' Episode Reward {np.sum(np.array(episode_reward)):10.3f} |'
                f' Detected {detected_at_step:10d} |'
            )

            writer.add_scalar(f'{env_string}/episode_reward', np.sum(episode_reward), global_step)
            writer.add_scalar(f'{env_string}/total_travel_time', np.sum(total_travel_time), global_step)
            writer.add_scalar(f'{env_string}/step_count', step_count, global_step)
            writer.add_scalar(f'{env_string}/detected_at_step', detected_at_step, global_step)

            episode += 1
            obs = env.reset()
            done = False
            truncated = False
            episode_reward = []
            total_travel_time = []
            step_count = 0
            detected_at_step = env.config['horizon']
            should_eval = episode % 10 == 0
            env_string = 'eval' if should_eval else 'env'

    weight_path = f'logs/{run_id}/weights'
    os.makedirs(weight_path, exist_ok=True)
    torch.save(attacker_model, f'{weight_path}/attacker_simple.pt')
    return attacker_model


def train_on_policy_simple_attacker(run_id, env, config, attacker_model):  # returns attacker strategy
    
    index = 'simple'

    writer = TBStatWriter(f'logs/{run_id}/attacker_{index}/')

    replay_buffer = TrajectoryExperience()

    done = False
    truncated = False
    obs = env.reset()
    episode_reward = []
    total_travel_time = []
    step_count = 0
    detected_at_step = env.config['horizon']
    episode = 0
    should_eval = episode % 10 == 0
    env_string = 'eval' if should_eval else 'env'

    for global_step in (pbar := tqdm(range(config['attacker_config']['training_steps']))):
        step_count += 1

        constructed_action, action, allocation, budget = attacker_model.forward_single(obs, deterministic=should_eval)
        # perturbed_edge_travel_times = env.get_travel_times_assuming_the_attack(constructed_action)
        # detected = bool(np.random.random() < sigmoid(budget[0], sl=2.0, th=30.0))
        # detected = bool(budget[0] > 30)
        detected = False

        if detected:
            detected_at_step = step_count
            value, cumulative_reward, steps, original_reward = __get_state_values_assuming_no_action(env, config['attacker_config']['gamma'], done)
            # value, cumulative_reward, steps, original_reward = np.zeros((4,)), np.zeros((4,)), 0, np.zeros((4,))
            # value = np.zeros((4,))
            # value = np.zeros((1, ))
            replay_buffer.add(
                obs, allocation, budget, action, value, obs, True, False)
            episode_reward.append(cumulative_reward)
            total_travel_time.extend(original_reward)
            step_count += steps
            writer.add_scalar(f'{env_string}/detected_value', np.sum(value), global_step)
            done = True
        else:
            next_obs, reward, done, info = env.step(
                constructed_action
            )
            truncated = info.get('TimeLimit.truncated', False)
            total_travel_time.append(info.get('original_reward'))
            replay_buffer.add(obs, allocation, budget,
                              action, reward, next_obs,
                              done, truncated)
            obs = next_obs
            episode_reward.append(reward)

        writer.add_scalar(f'env/buffer_size', replay_buffer.size(), global_step)

        if replay_buffer.size() == 256:
            with torch.autograd.set_detect_anomaly(True):
                observations, allocations, budgets, actions, rewards, next_observations, dones, truncateds = replay_buffer.get_experiences()
                stats = attacker_model.update(observations, allocations, budgets, actions, rewards,
                                              next_observations, dones, truncateds)
                replay_buffer.reset()
                writer.add_stats(stats, global_step)

        writer.add_scalar(f'{env_string}/attacker_budget', sum(constructed_action), global_step)
        writer.add_histogram(f'{env_string}/attacker_allocation', allocation, global_step)

        # TODO if this returns true, we should reset the experience replay buffer
        # attacker_model.attacker.iterative_scheduler.step()

        if done or truncated:
            pbar.set_description(
                f'Training Attacker |'
                f' ep: {episode} |'
                f' Episode Reward {np.sum(np.array(episode_reward)):10.3f} |'
                f' Detected {detected_at_step:10d} |'
            )

            writer.add_scalar(f'{env_string}/episode_reward', np.sum(episode_reward), global_step)
            writer.add_scalar(f'{env_string}/total_travel_time', np.sum(total_travel_time), global_step)
            writer.add_scalar(f'{env_string}/step_count', step_count, global_step)
            writer.add_scalar(f'{env_string}/detected_at_step', detected_at_step, global_step)

            episode += 1
            obs = env.reset()
            done = False
            truncated = False
            episode_reward = []
            total_travel_time = []
            step_count = 0
            detected_at_step = env.config['horizon']
            should_eval = episode % 10 == 0
            env_string = 'eval' if should_eval else 'env'

    weight_path = f'logs/{run_id}/weights'
    os.makedirs(weight_path, exist_ok=True)
    torch.save(attacker_model, f'{weight_path}/attacker_simple.pt')
    return attacker_model


if __name__ == '__main__':
    env = DynamicMultiAgentTransportationNetworkEnvironment(dict(
        network=dict(
            method='network_file',
            city='SiouxFalls',
            # method='edge_list',
            # file='GRE-3x3-0.5051-0.1111-20240416143549202501_default',
            # file='GRE-3x2-0.5051-0.1111-20240402180532473783_low',
            # file='GRE-2x2-0.5051-0.1111-20240415140456980176_default',
            randomize_factor=0.00,
        ),
        horizon=50,
        render_mode=None,
        congestion=True,
        rewarding_rule='proportional',
        # rewarding_rule='mixed',
        reward_multiplier=1.0,
        n_components=10,
    ))

    config = dict(
        attacker_config=dict(
            buffer_size=50_000,
            batch_size=64,
            training_steps=1024 * 1024,
            gamma=0.99,
        )
    )

    # attacker_model = NoisyAttacker(
    #     NoBudgetAttacker(
    #         'NoBudgetAttacker',
    #         env.edge_component_mapping,
    #         allocator=TD3NoBudgetAllocator(
    #             env.edge_component_mapping,
    #             5,
    #             config['attacker_config']['high_level']['critic_lr'],
    #             config['attacker_config']['high_level']['actor_lr'],
    #             config['attacker_config']['tau'],
    #             config['attacker_config']['gamma'],
    #             config['attacker_config']['target_noise_scale'],
    #             config['attacker_config']['actor_update_steps'],
    #         ),
    #         component=MATD3Component(
    #             env.edge_component_mapping,
    #             5,
    #             config['attacker_config']['low_level']['critic_lr'],
    #             config['attacker_config']['low_level']['actor_lr'],
    #             config['attacker_config']['tau'],
    #             config['attacker_config']['gamma'],
    #             config['attacker_config']['target_noise_scale'],
    #             config['attacker_config']['actor_update_steps'],
    #         ),
    #         iterative_scheduler=LevelTrainingScheduler(['allocator', 'component'], [1, 1])
    #         # iterative_scheduler=SimultaneousTrainingScheduler()
    #     ),
    #     budget_noise=GaussianNoiseDecay(0.0, 5.0, 0.01, 10_000),
    #     allocation_noise=GaussianNoiseDecay(0.0, 0.05, 0.005, 10_000),
    #     action_noise=GaussianNoiseDecay(0.0, 0.00, 0.0005, 10_000)
    # )

    free_flow_times = nx.get_edge_attributes(env.base, 'free_flow_time')
    env_adj = [free_flow_times[e] for e in env.base.edges()]
    env_adj = np.expand_dims(env_adj, axis=1)

    # attacker_model = NoisyAttacker(
    #     NoBudgetTD3Attacker(
    #         env.edge_component_mapping,
    #         5,
    #         env_adj,
    #         0.0001,
    #         0.01,
    #         0.99,
    #         0.001,
    #         0.01,
    #         2
    #     ),
    #     budget_noise=GaussianNoiseDecay(0.0, 5.0, 0.01, 10_000),
    #     allocation_noise=ZeroNoise(),
    #     action_noise=GaussianNoiseDecay(0.0, 0.5, 0.0005, 10_000)
    # )

    # attacker_model = EpsilonGreedyAttacker(
    #     NoisyAttacker(
    #         FixedBudgetNetworkedWideTD3(
    #             env_adj,
    #             env.edge_component_mapping,
    #             FixedBudgeting(30.0),
    #             6,
    #             0.0001,
    #             0.001,
    #             0.99,
    #             0.001,
    #             2,
    #             0.00
    #         ),
    #         budget_noise=ZeroNoise(),
    #         allocation_noise=ZeroNoise(),
    #         action_noise=GaussianNoiseDecay(0.0, 0.2, 0.0005, 20_000)
    #     ),
    #     epsilon=DecayEpsilon(0.4, 0.005, 40_000)
    # )

    self_edge_component_mapping = [
        [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14, 15, 19, 23, 31, 35],
        [13, 16, 17, 18, 20, 21, 22, 24, 25, 26, 29, 32, 43, 47, 48, 50, 51, 52, 54, 55, 60],
        [7, 10, 27, 33, 34, 36, 37, 38, 39, 40, 42, 44, 66, 70, 71, 73, 74, 76],
        [28, 30, 41, 45, 46, 49, 53, 57, 58, 59, 56, 61, 62, 63, 64, 65, 67, 68, 69, 72, 75]
    ]
    for i in range(len(self_edge_component_mapping)):
        for j in range(len(self_edge_component_mapping[i])):
            self_edge_component_mapping[i][j] -= 1

    attacker_model = FixedBudgetNetworkedWidePPO(
        self_edge_component_mapping,
        FixedBudgeting(30.0),
        5,
        0.0001,
        0.0001,
        0.3,
        32,
        0.99,
        0.95,
        0.5,
        0.05,
        10,
        True,
    )

    # attacker_model = Attacker(
    #     'HierarchicalPPOAttacker',
    #     env.edge_component_mapping,
    #     FixedBudgeting(30.0),
    #     # PPOAllocator(
    #     #     env.edge_component_mapping,
    #     #     5,
    #     #     0.0001,
    #     #     0.0001,
    #     #     0.99,
    #     #     0.95,
    #     #     0.3,
    #     #     None,
    #     #     0.5,
    #     #     0.05,
    #     #     10,
    #     #     32,
    #     #     None,
    #     #     0.0,
    #     #     normalize_advantages=True
    #     # ),
    #     ProportionalAllocator(),
    #     CoPPOComponent(
    #         env_adj,
    #         env.edge_component_mapping,
    #         5,
    #         0.0003,
    #         0.0003,
    #         0.99,
    #         0.95,
    #         0.3,
    #         0.2,
    #         None,
    #         0.5,
    #         0.01,
    #         10,
    #         16,
    #         None,
    #         True
    #     ),
    #     SimultaneousTrainingScheduler()
    # )

    print(attacker_model)

    train_on_policy_simple_attacker(f'multi/{datetime.now().strftime("%Y%m%d%H%M%S%f")}', env, config, attacker_model)
