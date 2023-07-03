import logging
import sys

from tqdm import tqdm

from attack_heuristics import PostProcessHeuristic, GreedyRiderVector, Zero
from transport_env.MultiAgentNetworkEnv import MultiAgentTransportationNetworkEnvironment

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)

    config = dict(
        network=dict(
            method='network_file',
            # city='Anaheim',
            city='SiouxFalls',
        ),
        # network=dict(
        #     method='generate',
        #     type='grid',
        #     width=7,
        #     height=7,
        # ),
        trips=dict(
            type='demand_file',
            demand_file='Sirui/traffic_data/sf_demand.txt',
            strategy='random',
            count=10
        ),
        # trips=dict(
        #     type='deterministic',
        #     count=10,
        # ),
        horizon=50,
        epsilon=30,
        norm=1,
        frac=0.5,
        num_sample=50,
        render_mode=None,
        reward_multiplier=1.0,
        congestion=True,
        # rewarding_rule='travel_time_increased',
        rewarding_rule='step_count',
        repeat=100,
        observation_type='vector',
        n_components=4,
        norm_penalty_coeff=1.0,
        capacity_divider=10000
    )

    env = MultiAgentTransportationNetworkEnvironment(config)

    # env.show_base_graph()

    # heuristic = PostProcessHeuristic(
    #     GreedyRiderVector(
    #         config['epsilon'],
    #         config['norm']
    #     )
    # )

    heuristic = Zero(
        (env.base.number_of_edges(), )
    )

    for episode in tqdm(range(config['repeat'])):

        obs = env.reset()['feature_vector']
        done = False
        truncated = False
        steps = 0
        norm_penalty_episode = 0

        while not done and not truncated:
            action = heuristic.predict(obs)
            obs, reward, done, info = env.step(action)
            obs = obs['feature_vector']
            truncated = info.get('TimeLimit.truncated', False)
            norm_penalty_episode += info['norm_penalty']
            steps += 1

            edge_travel_times = info['perturbed_edge_travel_times']

