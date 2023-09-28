import tqdm


from rl_trainer import train_single
from multiprocessing import Pool

if __name__ == '__main__':

    params = []

    for env_randomize_factor in [0.001]:
        for allocator_actor_lr in [0.0001, 0.00001, 0.000001]:
            for allocator_critic_lr in [0.0003, 0.0001, 0.003, 0.00001]:
                for allocator_gamma in [0.99]:
                    for allocator_lam in [0.95]:
                        for allocator_epsilon in [0.01, 0.1, 0.2, 0.3, 0.5]:
                            for allocator_entropy_coeff in [0.01, 0.1, 0.2, 0.5]:
                                for allocator_value_coeff in [0.5]:
                                    for allocator_n_updates in [1, 2, 5, 10, 20]:
                                        for allocator_policy_grad_clip in [None]:
                                            for allocator_batch_size in [64, 128]:
                                                for allocator_clip_range_vf in [None]:
                                                    for n_steps in [128, 512, 1024]:
                                                        for greedy_epsilon in [0.0, 0.5, 0.9]:
                                                            for allocator_kl_coeff in [0.0, 0.1, 0.01, 0.05]:
                                                                for allocator_normalize_advantages in [False, True]:
                                                                    for _ in range(1):
                                                                        params.append((
                                                                            n_steps,
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
                                                                            allocator_clip_range_vf,
                                                                            allocator_kl_coeff,
                                                                            allocator_normalize_advantages,
                                                                            greedy_epsilon,
                                                                            False
                                                                        ))

    print(f'Running {len(params)} experiments')

    with Pool(64) as pool:
        pool.starmap(
            train_single, tqdm.tqdm(params, total=len(params))
        )