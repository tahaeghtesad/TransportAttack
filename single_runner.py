from rl_trainer import train_single

if __name__ == '__main__':

    for i in train_single(
        env_randomize_factor=0.001,
        allocator_actor_lr=0.0001,
        allocator_critic_lr=0.0003,
        allocator_gamma=0.95,
        allocator_lam=0.0,
        allocator_epsilon=0.01,
        allocator_entropy_coeff=0.1,
        allocator_value_coeff=0.1,
        allocator_n_updates=10,
        allocator_policy_grad_clip=None,
        allocator_batch_size=128,
        allocator_max_concentration=50.0,
        allocator_clip_range_vf=None,
        greedy_epsilon=0.0,
        log_stdout=True,
    ):
        print(i)