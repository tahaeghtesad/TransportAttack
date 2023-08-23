import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils import tensorboard as tb
from tqdm import tqdm

import util.rl.exploration
from attack_heuristics import Random
from models.torch.maddpg import MADDPGModel
from transport_env.MultiAgentNetworkEnv import MultiAgentTransportationNetworkEnvironment
from util.rl.experience_replay import ExperienceReplay

if __name__ == '__main__':
    run_id = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    writer = tb.SummaryWriter(f'logs/{run_id}')
    os.makedirs(f'logs/{run_id}/weights')

    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    logger = logging.getLogger('main')

    config = dict(
        env_config=dict(
            network=dict(
                method='network_file',
                city='SiouxFalls',
                # city='Anaheim',
            ),
            # network=dict(
            #     method='generate',
            #     type='grid',
            #     width=7,
            #     height=7,
            # ),
            # network=dict(
            #     method='generate',
            #     type='line',
            #     num_nodes=10,
            # ),
            # network=dict(
            #     method='generate',
            #     type='cycle',
            #     num_nodes=20,
            # ),
            horizon=50,
            epsilon=30,
            norm=1,
            frac=0.5,
            num_sample=20,
            render_mode=None,
            # reward_multiplier=0.00001,
            reward_multiplier=1.0,
            congestion=True,
            trips=dict(
                type='trips_file',
                randomize_factor=0.002,
            ),
            rewarding_rule='step_count',
            # norm_penalty_coeff=0.00001,
            norm_penalty_coeff=1.0,
            n_components=4,
        )
    )

    with open(f'logs/{run_id}/config.json', 'w') as fd:
        json.dump(config, fd, indent=4)

    logger.info(config)

    env = MultiAgentTransportationNetworkEnvironment(config['env_config'])
    # env.show_base_graph()

    device = torch.device('cpu')
    logger.info(device)

    model = MADDPGModel(env, config, device)
    logger.info(model)

    buffer = ExperienceReplay(50_000, 128)

    num_episodes = 100_000

    random = Random(
        action_shape=sum([len(c) for c in env.edge_component_mapping]),
        norm=config['env_config']['norm'],
        epsilon=config['env_config']['epsilon'],
        frac=config['env_config']['frac'],
        selection='discrete'
    )

    epsilon = util.rl.exploration.DecayEpsilon(
        epsilon_start=0.3,
        epsilon_end=0.1,
        epsilon_decay=10_000
    )

    noise = util.rl.exploration.OUActionNoise(
        theta=0.15,
        mean=0.0,
        std_deviation=0.1,
        dt=0.01,
        target_scale=0.005,
        anneal=20_000,
        shape=sum([len(c) for c in env.edge_component_mapping])
    )

    pbar = tqdm(total=num_episodes)
    global_step = 0
    total_samples = 0

    for episode in range(num_episodes):

        should_test = (episode + 1) % 100 == 0

        state = env.reset()
        done = False
        truncated = False
        rewards = 0
        component_rewards = np.zeros(env.n_components)
        step = 0
        discounted_reward = 0
        norm_penalty = 0
        original_reward = 0
        while not done and not truncated:
            if not should_test and epsilon():
                action = random.predict(state)
            else:
                action = model.forward(state)

            if not should_test:
                action += noise()

            next_state, reward, done, info = env.step(action)
            truncated = info.get("TimeLimit.truncated", False)
            buffer.add(state, action, reward, next_state, done)
            norm_penalty += info.get('norm_penalty')
            original_reward += info.get('original_reward')
            state = next_state
            rewards += sum(reward)
            component_rewards += reward
            discounted_reward += sum(reward) * (0.97 ** step)
            step += 1
            total_samples += 1

        target_cat = 'test' if should_test else 'env'
        writer.add_scalar(f'{target_cat}/cumulative_reward', rewards, global_step)
        writer.add_scalar(f'{target_cat}/discounted_reward', discounted_reward, global_step)
        writer.add_scalar(f'{target_cat}/episode_length', step, global_step)
        writer.add_scalar(f'{target_cat}/norm_penalty', norm_penalty / step, global_step)
        writer.add_scalar(f'{target_cat}/original_reward', original_reward, global_step)

        for c in range(env.n_components):
            writer.add_scalar(f'{target_cat}/component_reward/{c}', component_rewards[c], global_step)

        for _ in range(4):

            if buffer.size() >= 128:
                states, actions, rewards, next_states, dones = buffer.sample()
                stats = model.update_multi_agent(states, actions, next_states, rewards, dones)

                # writer.add_scalar('model/actor_q', stats['actor_q'], global_step)
                if type(stats['loss']) == list:
                    for c in range(env.n_components):
                        writer.add_scalar(f'model/loss/{c}', stats['loss'][c], global_step)
                        writer.add_scalar(f'model/r2/{c}', max(stats['r2'][c], -1), global_step)

                        writer.add_scalar(f'q/max_q/{c}', stats['max_q'][c], global_step)
                        writer.add_scalar(f'q/mean_q/{c}', stats['mean_q'][c], global_step)
                        writer.add_scalar(f'q/min_q/{c}', stats['min_q'][c], global_step)
                else:
                    writer.add_scalar('model/loss', stats['loss'], global_step)
                    writer.add_scalar('model/r2', max(stats['r2'], -1), global_step)

                    writer.add_scalar('q/max_q', stats['max_q'], global_step)
                    writer.add_scalar('q/mean_q', stats['mean_q'], global_step)
                    writer.add_scalar('q/min_q', stats['min_q'], global_step)

                writer.add_scalar('exploration/epsilon', epsilon.get_current_epsilon(), global_step)
                writer.add_scalar('exploration/noise', noise.get_current_noise(), global_step)

                writer.add_scalar('experiences/buffer_size', buffer.size(), global_step)
                writer.add_scalar('experiences/total_samples', total_samples, global_step)

                pbar.set_description(
                    f'QLoss {["%.2f" % c for c in stats["loss"]]} | ' if type(stats['loss']) == list else f'QLoss {stats["loss"]:.2f} | '
                    f'R2 {["%.2f" % c for c in stats["r2"]]} | ' if type(stats['r2']) == list else f'R2 {stats["r2"]:.2f} | '
                    f'MaxQ {["%.2f" % c for c in stats["max_q"]]} | ' if type(stats['max_q']) == list else f'MaxQ {stats["max_q"]:.2f} | '
                    f'MeanQ {["%.2f" % c for c in stats["mean_q"]]} | ' if type(stats['mean_q']) == list else f'MeanQ {stats["mean_q"]:.2f} | '
                    f'MinQ {["%.2f" % c for c in stats["min_q"]]} | ' if type(stats['min_q']) == list else f'MinQ {stats["min_q"]:.2f} | '
                    f'Episode {episode} | '
                    f'Len {step} | '
                    f'CumReward {rewards:.2f} | '
                    f'DisReward {discounted_reward:.3f} | '
                    f'Eps {epsilon.get_current_epsilon():.3f} | '
                    f'Noise {noise.get_current_noise():.2f} | '
                    f'ReplayBuffer {buffer.size()}'
                )

            pbar.update(1)
            global_step += 1

            if global_step % 1000 == 0:
                torch.save(model.state_dict(), f'logs/{run_id}/weights/model_{global_step}.pt')
