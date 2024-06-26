import os
import random
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from models import CustomModule
from models.agents.epsilon_greedy import EpsilonGreedyDetector
from models.agents.heuristics.attackers.attackers import Zero
from models.agents.heuristics.attackers.component import GreedyComponent
from models.agents.noisy_agent import NoisyAttacker
from models.agents.rl_agents.attackers.allocators.allocators import TD3NoBudgetAllocator
from models.agents.rl_agents.attackers.component.maddpg_component import MATD3Component
from models.agents.rl_agents.attackers.mixed_td3_attacker import MixedTD3Attacker
from models.agents.rl_agents.attackers.rl_attackers import BaseAttacker, NoBudgetAttacker
from models.agents.rl_agents.defenders.detectors.detectors import BaseDetector, DoubleDQNDetector
from models.exploration.epsilon import DecayEpsilon
from models.exploration.noise import OUActionNoise, ZeroNoise, GaussianNoiseDecay
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment
from util.math import solve_lp, sigmoid
from util.rl.experience_replay import ExperienceReplay, BasicExperienceReplay
from util.scheduler import LevelTrainingScheduler, SimultaneousTrainingScheduler
from util.torch.writer import TBStatWriter


class Trainer(CustomModule):

    def __init__(self, config, env: DynamicMultiAgentTransportationNetworkEnvironment) -> None:
        super().__init__('DOTrainer')
        self.env = env
        self.config = config

        self.attacker_strategy_sets: List[BaseAttacker] = []
        self.defender_strategy_sets: List[BaseDetector] = []

        # Rows are the defender, columns are the attacker. Values are the payoffs for the defender.
        self.payoff_table = []

    def __get_state_values_assuming_no_action(self, done):
        gamma = self.config['attacker_config']['gamma'] * 0

        truncated = False
        immediate_rewards = []
        original_reward = []
        step_count = -1

        while not done and not truncated:
            step_count += 1
            action = np.zeros((self.env.base.number_of_edges(),))
            obs, reward, done, info = self.env.step(action)
            truncated = info.get('TimeLimit.truncated', False)
            original_reward.append(info.get('original_reward'))
            immediate_rewards.append(reward)

        immediate_rewards = np.array(immediate_rewards)
        state_value = np.zeros(immediate_rewards.shape[1])
        for i in range(immediate_rewards.shape[0]):
            state_value += gamma ** i * immediate_rewards[i]

        return state_value, np.sum(immediate_rewards, axis=0), step_count, original_reward

    def train_attacker(self, probabilities) -> BaseAttacker:  # returns attacker strategy
        assert len(self.defender_strategy_sets) == len(probabilities)
        index = len(self.attacker_strategy_sets)

        writer = TBStatWriter(f'logs/{self.config["run_id"]}/attacker_{index}/')

        replay_buffer = ExperienceReplay(self.config['attacker_config']['buffer_size'],
                                         self.config['attacker_config']['batch_size'])
        self.env.config['rewarding_rule'] = 'mixed'

        attacker_model = NoisyAttacker(
            MixedTD3Attacker(
                self.env.edge_component_mapping,
                5,
                self.config['attacker_config']['low_level']['actor_lr'],
                self.config['attacker_config']['low_level']['critic_lr'],
                self.config['attacker_config']['high_level']['actor_lr'],
                self.config['attacker_config']['tau'],
                self.config['attacker_config']['gamma'],
                self.config['attacker_config']['actor_update_steps'],
                self.config['attacker_config']['target_noise_scale'],
            ),
            budget_noise=OUActionNoise(0.0, 5.0, 0.0, 5_000),
            allocation_noise=ZeroNoise(),
            action_noise=ZeroNoise()
        )

        done = False
        truncated = False
        obs = self.env.reset()
        episode_reward = []
        total_travel_time = []
        step_count = 0
        detector = self.defender_strategy_sets[
            np.random.choice(len(self.defender_strategy_sets), p=probabilities)]
        detected_at_step = self.env.config['horizon']
        episode = 0
        should_eval = episode % 10 == 0
        env_string = 'eval' if should_eval else 'env'

        for global_step in (pbar := tqdm(range(self.config['attacker_config']['training_steps']))):
            step_count += 1

            constructed_action, action, allocation, budget = attacker_model.forward_single(obs,
                                                                                           deterministic=should_eval)
            perturbed_edge_travel_times = self.env.get_travel_times_assuming_the_attack(constructed_action)
            detected = detector.forward_single(perturbed_edge_travel_times, True)

            if detected:
                detected_at_step = step_count
                value, cumulative_reward, steps, original_reward = self.__get_state_values_assuming_no_action(done)
                replay_buffer.add(
                    obs, allocation, budget, action, value, obs, True, False)
                episode_reward.append(cumulative_reward)
                total_travel_time.extend(original_reward)
                step_count += steps
                writer.add_scalar(f'{env_string}/detected_value', np.sum(value), global_step)
                done = True
            else:
                next_obs, reward, done, info = self.env.step(
                    constructed_action
                )
                truncated = info.get('TimeLimit.truncated', False)
                total_travel_time.append(info.get('original_reward'))
                replay_buffer.add(obs, allocation, budget, action, reward, next_obs, done, truncated)
                obs = next_obs
                episode_reward.append(reward)

            writer.add_scalar(f'env/buffer_size', replay_buffer.size(), global_step)

            if replay_buffer.size() > self.config['attacker_config']['batch_size']:
                observations, allocations, budgets, actions, rewards, next_observations, dones, truncateds = replay_buffer.get_experiences()
                stats = attacker_model.update(observations, allocations, budgets, actions, rewards,
                                              next_observations, dones, truncateds)
                writer.add_stats(stats, global_step)

            writer.add_scalar(f'{env_string}/attacker_budget', budget, global_step)

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
                obs = self.env.reset()
                done = False
                truncated = False
                episode_reward = []
                total_travel_time = []
                step_count = 0
                should_eval = episode % 10 == 0
                detector = self.defender_strategy_sets[
                    np.random.choice(len(self.defender_strategy_sets), p=probabilities)]
                detected_at_step = self.env.config['horizon']
                should_eval = episode % 10 == 0
                env_string = 'eval' if should_eval else 'env'

        weight_path = f'logs/{self.config["run_id"]}/weights'
        os.makedirs(weight_path, exist_ok=True)
        torch.save(attacker_model, f'{weight_path}/attacker_{index}.pt')
        return attacker_model

    def train_detector(self, probabilities) -> BaseDetector:  # returns defender strategy
        assert len(self.attacker_strategy_sets) == len(probabilities)
        detector_model = EpsilonGreedyDetector(
            DoubleDQNDetector(
                self.env.base.number_of_edges(),
                1,
                self.config['detector_config']['gamma'],
                self.config['detector_config']['tau'],
                self.config['detector_config']['lr'],
            ),
            epsilon=DecayEpsilon(
                self.config['detector_config']['epsilon']['start'],
                self.config['detector_config']['epsilon']['end'],
                self.config['detector_config']['epsilon']['decay']
            )
        )
        index = len(self.defender_strategy_sets)
        writer = TBStatWriter(f'logs/{self.config["run_id"]}/defender_{index}/')
        replay_buffer = BasicExperienceReplay(
            self.config['detector_config']['buffer_size'],
            self.config['detector_config']['batch_size'])

        self.env.config['rewarding_rule'] = 'proportional'

        attacker_present = random.random() < self.config['detector_config']['attacker_present_probability']
        zero_attacker = Zero(self.env.edge_component_mapping)
        attacker_model = self.attacker_strategy_sets[
            np.random.choice(len(self.attacker_strategy_sets), p=probabilities)]
        done = False
        truncated = False
        episode = 0
        obs = self.env.reset()
        step_count = 0
        episode_rewards = []
        total_travel_time = []
        detected_at_step = self.env.config['horizon'] if attacker_present else -1

        attacker_action = attacker_model.forward_single(obs, True)[0] if attacker_present else \
            zero_attacker.forward_single(obs, True)[0]

        for global_step in (pbar := tqdm(range(self.config['detector_config']['training_steps']))):

            step_count += 1

            perturbed_edge_travel_times = self.env.get_travel_times_assuming_the_attack(attacker_action)
            detected = detector_model.forward_single(perturbed_edge_travel_times, False)
            detector_penalty = 0

            attack_correctly_detected = detected and attacker_present
            false_positive = detected and not attacker_present

            if attack_correctly_detected:  # We neutralize the attack
                attacker_present = False
                detected_at_step = step_count
            if false_positive:  # We penalize the detector
                detector_penalty = self.config['detector_config']['rho']

            obs, reward, done, info = self.env.step(
                attacker_action if attacker_present else zero_attacker.forward_single(obs, True)[0]
            )

            truncated = info.get('TimeLimit.truncated', False)
            travel_time = info.get('original_reward')
            detector_reward = - sum(reward) - detector_penalty

            episode_rewards.append(detector_reward)
            total_travel_time.append(travel_time)

            next_attacker_action, next_attacker_normalized_action, next_attacker_allocation, next_attacker_budget = attacker_model.forward_single(
                obs, True) if attacker_present else zero_attacker.forward_single(obs, True)
            next_detector_observation = self.env.get_travel_times_assuming_the_attack(next_attacker_action)

            replay_buffer.add(
                perturbed_edge_travel_times,
                detected,
                detector_reward,
                next_detector_observation,
                done,
                truncated
            )

            writer.add_scalar('env/attacker_budget', next_attacker_budget, global_step)

            attacker_action = next_attacker_action

            if replay_buffer.size() > self.config['detector_config']['batch_size']:
                states, actions, rewards, next_states, dones, truncateds = replay_buffer.get_experiences()
                stats = detector_model.update(states, actions, next_states, rewards, dones)
                writer.add_stats(stats, global_step)

            if done or truncated:
                writer.add_scalar('env/detected_at_step', detected_at_step, global_step)
                writer.add_scalar('env/total_travel_time', np.sum(total_travel_time), global_step)
                writer.add_scalar('env/episode_reward', np.sum(episode_rewards), global_step)
                writer.add_scalar('env/step_count', step_count, global_step)
                writer.add_scalar('env/buffer_size', replay_buffer.size(), global_step)
                pbar.set_description(
                    f'Training Detector | ep: {episode} | Rewards {np.sum(episode_rewards):10.3f} | detected {detected_at_step:10d} |')

                episode += 1
                obs = self.env.reset()
                done = False
                truncated = False
                episode_rewards = []
                total_travel_time = []
                step_count = 0
                attacker_present = random.random() < self.config['detector_config']['attacker_present_probability']
                attacker_model = self.attacker_strategy_sets[
                    np.random.choice(len(self.attacker_strategy_sets), p=probabilities)]
                zero_attacker = Zero(self.env.edge_component_mapping)
                attacker_action = attacker_model.forward_single(obs, True)[0] if attacker_present else \
                    zero_attacker.forward_single(obs, True)[0]
                detected_at_step = self.env.config['horizon'] if attacker_present else -1

        weight_path = f'logs/{self.config["run_id"]}/weights'
        os.makedirs(weight_path, exist_ok=True)
        torch.save(detector_model, f'{weight_path}/defender_{index}.tar')
        return detector_model

    def play(self, attacker_model: BaseAttacker, detector_model: BaseDetector) -> (float, float):

        detector_rewards = []
        attacker_rewards = []
        global_step = 0
        detected_epsiodes = 0

        for episode in (pbar := tqdm(range(self.config['do_config']['testing_epochs']))):

            done = False
            truncated = False
            obs = self.env.reset()
            step_count = 0
            attacker_present = random.random() < self.config['detector_config']['attacker_present_probability']

            while not done and not truncated:
                step_count += 1
                global_step += 1

                attacker_action = attacker_model.forward_single(obs, True)[0] if attacker_present else np.zeros(
                    (self.env.base.number_of_edges(),))
                perturbed_edge_travel_times = self.env.get_travel_times_assuming_the_attack(attacker_action)
                detected = detector_model.forward_single(perturbed_edge_travel_times, True)
                detector_penalty = 0

                if detected and attacker_present:
                    attacker_present = False
                    detected_epsiodes += 1
                if detected and not attacker_present:
                    detector_penalty = self.config['detector_config']['rho']

                obs, reward, done, info = self.env.step(attacker_action if attacker_present else np.zeros(
                    (self.env.base.number_of_edges(),)))

                truncated = info.get('TimeLimit.truncated', False)
                attacker_reward = sum(reward)
                detector_reward = - attacker_reward - detector_penalty

                attacker_rewards.append(attacker_reward)
                detector_rewards.append(detector_reward)
                self.logger.info(f'Detection Rate: {detected_epsiodes / (self.config["do_config"]["testing_epochs"])}')

        return np.mean(attacker_rewards), np.mean(detector_rewards)

    def get_attacker_payoff(self, attacker: BaseAttacker):  # returns a list of payoffs for the defender
        payoffs = []

        for i, classifier in enumerate(self.defender_strategy_sets):
            payoff = self.play(attacker, classifier)
            payoffs.append(payoff)

        self.store_payoff_table()

        return payoffs

    def get_defender_payoff(self, classifier: BaseDetector):  # returns a list of payoffs for the attacker
        payoffs = []

        for i, attacker in enumerate(self.attacker_strategy_sets):
            payoff = self.play(attacker, classifier)
            payoffs.append(payoff)

        self.store_payoff_table()

        return payoffs

    def store_payoff_table(self):
        os.makedirs(f'logs/{self.config["run_id"]}', exist_ok=True)
        np.savetxt(f'logs/{self.config["run_id"]}/payoff_table.csv', np.array(self.payoff_table), delimiter=',')

    def append_attacker_payoffs(self, attacker_payoffs: list):
        assert len(self.defender_strategy_sets) == len(attacker_payoffs)
        for i, payoff in enumerate(attacker_payoffs):
            self.payoff_table[i].append(payoff)

    def append_defender_payoffs(self, defender_payoffs: list):
        assert len(self.attacker_strategy_sets) == len(defender_payoffs)
        self.payoff_table.append(defender_payoffs)

    def solve_defender(self):
        return solve_lp(np.transpose(-np.array(self.payoff_table)))

    def solve_attacker(self):
        return solve_lp(np.array(self.payoff_table))
