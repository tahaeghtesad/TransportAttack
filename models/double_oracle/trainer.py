import logging
import random

import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from models.attack_heuristics import Random
from models.dl.classifier import Classifier
from models.dl.hmaddpg import MADDPGModel
from util.graphing import create_roc_curve
from util.math import solve_lp
from util.rl.experience_replay import ExperienceReplay
from util.rl.exploration import OUActionNoise, DecayEpsilon
from torch.utils import tensorboard as tb


class Trainer:

    def __init__(self, config, env, device) -> None:
        super().__init__()
        self.logger = logging.getLogger('Trainer')
        self.config = config
        self.env = env
        self.device = device

        self.attacker_strategy_sets = []
        self.defender_strategy_sets = []

        # Rows are the defender, columns are the attacker. Values are the payoffs for the defender.
        self.payoff_table = []

    def __get_state_values_assuming_no_action(self):
        gamma = self.config['rl_config']['gamma']

        done = False
        truncated = False
        immediate_rewards = []
        step_count = -1

        while not done and not truncated:
            step_count += 1
            action = np.zeros((self.env.base.number_of_edges(),))
            obs, reward, done, info = self.env.step(action)
            truncated = info.get('TimeLimit.truncated', False)
            immediate_rewards.append(reward)

        immediate_rewards = np.array(immediate_rewards)
        state_value = np.zeros(immediate_rewards.shape[1])
        for i in range(immediate_rewards.shape[0]):
            state_value += gamma ** i * immediate_rewards[i]

        return state_value, np.sum(immediate_rewards, axis=0), step_count

    def train_attacker(self, probabilities):  # returns attacker strategy
        assert len(self.defender_strategy_sets) == len(probabilities)
        index = len(self.attacker_strategy_sets)

        writer = tb.SummaryWriter(f'logs/{self.config["run_id"]}/attacker_{index}/')
        attacker_model = MADDPGModel(self.config, self.env, self.device)
        replay_buffer = ExperienceReplay(self.config['rl_config']['buffer_size'],
                                         self.config['rl_config']['batch_size'])
        episode_reward = np.zeros((4,))
        global_step = 0

        noise = OUActionNoise(
            theta=0.15,
            mean=0.0,
            std_deviation=self.config['rl_config']['noise']['std_deviation'],
            dt=0.01,
            target_scale=self.config['rl_config']['noise']['target_scale'],
            anneal=self.config['rl_config']['noise']['anneal'],
            shape=self.env.base.number_of_edges()
        )

        epsilon = DecayEpsilon(
            epsilon_start=self.config['rl_config']['exploration']['start'],
            epsilon_end=self.config['rl_config']['exploration']['end'],
            epsilon_decay=self.config['rl_config']['exploration']['decay'],
        )

        for episode in (pbar := tqdm(range(self.config['rl_config']['epochs']))):
            done = False
            truncated = False
            obs = self.env.reset()
            cumulative_rewards = []
            step_count = 0
            classifier = self.defender_strategy_sets[
                np.random.choice(len(self.defender_strategy_sets), p=probabilities)]
            detected_at_step = 1000
            detected = False

            while not done and not truncated and not detected:
                step_count += 1
                global_step += 1

                action, budget, magnitude = attacker_model.forward(obs)

                if epsilon():
                    action = Random(
                        (self.env.base.number_of_edges(),),
                        norm=1,
                        epsilon=np.exp(np.random.normal(1.1, 0.3)),
                        frac=0.5,
                        selection='discrete'
                    ).predict(obs)

                action += noise()
                action = np.clip(action, 0, None)

                true_action_budget = np.sum(action)

                perturbed_edge_travel_times = self.env.get_travel_times_assuming_the_attack(action)
                detected = classifier.forward_single(perturbed_edge_travel_times)
                # detected = False
                if detected:
                    detected_at_step = step_count
                    value, cumulative_reward, steps = self.__get_state_values_assuming_no_action()
                    replay_buffer.add(obs, action, np.append(value, [1.0]), obs, False)
                    cumulative_rewards.append(cumulative_reward)
                    step_count += steps
                    global_step += steps
                    writer.add_scalar('env/detected_value', np.sum(value), global_step)
                else:
                    next_obs, reward, done, info = self.env.step(
                        action
                    )

                    truncated = info.get('TimeLimit.truncated', False)
                    replay_buffer.add(obs, action, np.append(reward, [0]), next_obs, done)
                    obs = next_obs
                    cumulative_rewards.append(reward)

                writer.add_scalar('env/buffer_size', replay_buffer.size(), global_step)

                stats = []

                if replay_buffer.size() > self.config['rl_config']['batch_size']:
                    for _ in range(self.config['rl_config']['updates']):
                        states, actions, rewards, next_states, dones = replay_buffer.get_experiences()
                        stat = attacker_model.update_multi_agent(states, actions, next_states, rewards, dones)
                        stats.append(stat)

                    for i in range(self.env.n_components):
                        writer.add_scalar(f'loss/critic_{i}', np.mean([stat['q_loss'][i] for stat in stats]), global_step)
                        writer.add_scalar(f'min_q/critic_{i}', np.mean([stat['q_min'][i] for stat in stats]), global_step)
                        writer.add_scalar(f'max_q/critic_{i}', np.mean([stat['q_max'][i] for stat in stats]), global_step)
                        writer.add_scalar(f'mean_q/critic_{i}', np.mean([stat['q_mean'][i] for stat in stats]), global_step)
                        writer.add_scalar(f'r2/critic_{i}', np.mean([stat['q_r2'][i] for stat in stats]), global_step)
                        writer.add_scalar(f'a_loss/actor_{i}', np.mean([stat['a_loss'][i] for stat in stats]), global_step)

                    writer.add_scalar(f'high_level/loss', np.mean([stat['high_level_loss'] for stat in stats]), global_step)
                    writer.add_scalar(f'high_level/r2', np.mean([stat['high_level_r2'] for stat in stats]), global_step)
                    writer.add_scalar(f'high_level/min_q', np.mean([stat['high_level_min_q'] for stat in stats]), global_step)
                    writer.add_scalar(f'high_level/max_q', np.mean([stat['high_level_max_q'] for stat in stats]), global_step)
                    writer.add_scalar(f'high_level/mean_q', np.mean([stat['high_level_mean_q'] for stat in stats]), global_step)
                    writer.add_scalar(f'high_level/a_loss', np.mean([stat['high_level_a_loss'] for stat in stats]), global_step)

                writer.add_scalar(f'env/noise', noise.get_current_noise(), global_step)
                writer.add_scalar(f'env/epsilon', epsilon.get_current_epsilon(), global_step)
                writer.add_scalar(f'env/magnitude', magnitude, global_step)

                writer.add_scalar(f'env/true_action_budget', true_action_budget, global_step)

                for i in range(self.env.n_components):
                    writer.add_scalar(f'budgets/budget_{i}', np.sum(action[self.env.edge_component_mapping[i]]),
                                      global_step)
                    writer.add_scalar(f'budgets/vehicles_{i}',
                                      np.sum(obs[self.env.edge_component_mapping[i], 2]),
                                      global_step)

            pbar.set_description(
                f'Training Attacker |'
                f' ep: {episode} |'
                f' Rewards {episode_reward}:{np.sum(episode_reward):10.3f} |'
                f' Noise {noise.get_current_noise():10.3f} |'
                f' Exploration {epsilon.get_current_epsilon():10.3f} |'
                f' Detected {min(detected_at_step, self.config["env_config"]["horizon"]):10d} |'
            )

            episode_reward = np.sum(np.array(cumulative_rewards), axis=0)
            for i in range(self.env.n_components):
                writer.add_scalar(f'rewards/episode_reward_{i}', episode_reward[i] / self.config['env_config']['reward_multiplier'], global_step)
            writer.add_scalar(f'env/episode_reward', np.sum(episode_reward) / self.config['env_config']['reward_multiplier'], global_step)
            writer.add_scalar(f'env/step_count', step_count, global_step)
            writer.add_scalar(f'env/detected_at_step', min(detected_at_step, self.config['env_config']['horizon']), global_step)

        attacker_model.save(f'logs/{self.config["run_id"]}/weights/attacker_{index}.pt')
        return attacker_model

    def __get_classifier_data(self, probabilities, epochs):
        assert len(self.attacker_strategy_sets) == len(probabilities)

        edge_travel_times = []
        labels = []

        for episode in tqdm(range(epochs), desc='Collecting Data'):
            done = False
            truncated = False
            obs = self.env.reset()
            step_count = 0
            attacker_model = self.attacker_strategy_sets[
                np.random.choice(len(self.attacker_strategy_sets), p=probabilities)]

            while not done and not truncated:
                step_count += 1
                perturbed = False

                if random.random() < 0.4:
                    action, budget, magnitude = attacker_model.forward(obs)
                    perturbed = True
                else:
                    action = np.zeros((self.env.base.number_of_edges(),))

                obs, reward, done, info = self.env.step(action)

                truncated = info.get('TimeLimit.truncated', False)
                perturbed_edge_travel_times = info.get('perturbed_edge_travel_times')
                edge_travel_times.append(perturbed_edge_travel_times)
                labels.append(perturbed)

        return edge_travel_times, labels

    def train_classifier(self, probabilities):  # returns defender strategy
        assert len(self.attacker_strategy_sets) == len(probabilities)
        classifier = Classifier(self.config, self.device,
                                (self.env.base.number_of_edges(),))
        index = len(self.defender_strategy_sets)
        writer = tb.SummaryWriter(f'logs/{self.config["run_id"]}/defender_{index}/')

        edge_travel_times, labels = self.__get_classifier_data(probabilities, self.config['classifier_config']['collection_epochs'])

        global_step = 0
        for episode in (pbar := tqdm(range(self.config['classifier_config']['training_epochs']))):
            for loss in classifier.update_batched(edge_travel_times, labels):
                writer.add_scalar('defender/loss', loss['mean'], global_step)
                pbar.set_description(f'Training Classifier | loss: {loss["mean"]:10.3f}')
                global_step += 1

        classifier.save(f'logs/{self.config["run_id"]}/weights/classifier_{index}.pt')

        edge_travel_times, labels = self.__get_classifier_data(probabilities, self.config['do_config']['testing_epochs'])

        create_roc_curve(
            classifier.forward(edge_travel_times),
            labels,
            f'Classifier {index} ROC',
            f'logs/{self.config["run_id"]}/roc_curves/classifier_{index}.tikz',
            show=True
        )

        return classifier

    def play(self, attacker, defender):

        edge_travel_times = []
        labels = []
        predicted_labels = []
        episode_rewards = []

        for episode in range(self.config['do_config']['testing_epochs']):

            done = False
            truncated = False
            obs = self.env.reset()
            step_count = 0
            rewards = 0
            detected = False

            while not done and not truncated and not detected:
                step_count += 1
                if random.random() < 0.3:
                    action, budget, magnitude = attacker.forward(obs)
                    labels.append(1)
                else:
                    action = np.zeros((self.env.base.number_of_edges(),))
                    labels.append(0)

                perturbed_edge_travel_times = self.env.get_travel_times_assuming_the_attack(action)
                edge_travel_times.append(perturbed_edge_travel_times)
                detected = defender.forward_single(perturbed_edge_travel_times)
                predicted_labels.append(detected)

                if not detected:
                    obs, reward, done, info = self.env.step(action)
                    truncated = info.get('TimeLimit.truncated', False)
                    rewards += sum(reward)
                else:
                    value, cumulative_reward, steps = self.__get_state_values_assuming_no_action()
                    rewards += sum(cumulative_reward)
                    step_count += steps

            episode_rewards.append(rewards)

        average_reward = np.mean(episode_rewards)
        cm = confusion_matrix(labels, predicted_labels)
        false_positive = cm[0][1] / sum(cm[0])
        payoff = - average_reward - false_positive * self.config['classifier_config']['chi']
        self.logger.info(
            f'Attacker  vs. Defender |'
            f' Payoff: {payoff} |'
            f' FPR: {false_positive} |'
            f' TPR: {cm[1][1] / sum(cm[1])} |'
            f' FNR: {cm[1][0] / sum(cm[1])} |'
            f' TNR: {cm[0][0] / sum(cm[0])} |'
        )

        return payoff

    def get_attacker_payoff(self, attacker):  # returns a list of payoffs for the defender
        payoffs = []

        for i, classifier in enumerate(self.defender_strategy_sets):
            payoff = self.play(attacker, classifier)
            payoffs.append(payoff)

        return payoffs

    def get_defender_payoff(self, classifier):  # returns a list of payoffs for the attacker
        payoffs = []

        for i, attacker in enumerate(self.attacker_strategy_sets):
            payoff = self.play(attacker, classifier)
            payoffs.append(payoff)

        return payoffs

    def append_attacker_payoffs(self, attacker_payoffs):
        assert len(self.defender_strategy_sets) == len(attacker_payoffs)
        for i, payoff in enumerate(attacker_payoffs):
            self.payoff_table[i].append(payoff)

    def append_defender_payoffs(self, defender_payoffs):
        assert len(self.attacker_strategy_sets) == len(defender_payoffs)
        self.payoff_table.append(defender_payoffs)

    def solve_defender(self):
        return solve_lp(np.transpose(-np.array(self.payoff_table)))

    def solve_attacker(self):
        return solve_lp(np.array(self.payoff_table))