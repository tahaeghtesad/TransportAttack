import tqdm

from rl_trainer import train_single
from multiprocessing import Pool

if __name__ == '__main__':

    params = []

    # env_randomize_factor,
    # allocator_actor_lr,
    # allocator_critic_lr,
    # allocator_gamma,
    # allocator_lam,
    # allocator_epsilon,
    # allocator_entropy_coeff,
    # allocator_value_coeff,
    # allocator_n_updates,
    # allocator_policy_grad_clip,
    # allocator_batch_size,
    # allocator_max_concentration,
    # allocator_clip_range_vf,

    for env_randomize_factor in [0.001]:
        for allocator_actor_lr in [0.0003, 0.0001, 0.00001, 0.000001]:
            for allocator_critic_lr in [0.0003]:
                for allocator_gamma in [0.97]:
                    for allocator_lam in [0.95]:
                        for allocator_epsilon in [0.3]:
                            for allocator_entropy_coeff in [0.01, 0.1, 0.2, 0.5]:
                                for allocator_value_coeff in [0.5]:
                                    for allocator_n_updates in [10]:
                                        for allocator_policy_grad_clip in [0.05, None]:
                                            for allocator_batch_size in [128]:
                                                for allocator_max_concentration in [50.0]:
                                                    for allocator_clip_range_vf in [10.0, None]:
                                                        for _ in range(5):
                                                            params.append((
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
                                                                False  # log_stdout
                                                            ))

    print(f'Running {len(params)} experiments')

    with Pool(8) as pool:
        pool.starmap(
            train_single, tqdm.tqdm(params, total=len(params))
        )