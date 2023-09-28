from rl_trainer import train_single

if __name__ == '__main__':

    for i in train_single(
        n_steps=64,
        env_randomize_factor=0.001,
        allocator_actor_lr=0.00001,
        allocator_critic_lr=0.0003,
        allocator_gamma=0.99,
        allocator_lam=0.95,
        allocator_epsilon=0.1,
        allocator_entropy_coeff=0.1,
        allocator_value_coeff=0.5,
        allocator_n_updates=1,
        allocator_policy_grad_clip=None,
        allocator_batch_size=64,
        allocator_clip_range_vf=None,
        allocator_kl_coeff=0.0,
        allocator_normalize_advantages=True,
        # component_actor_lr=0.0003,
        # component_critic_lr=0.0003,
        # component_gamma=0.97,
        # component_lam=0.95,
        # # component_epsilon=0.3,
        # component_epsilon=5.0,
        # component_entropy_coeff=0.00,
        # component_value_coeff=0.5,
        # component_n_updates=10,
        # component_policy_grad_clip=None,
        # component_batch_size=128,
        # component_max_concentration=50.0,
        # component_clip_range_vf=None,
        greedy_epsilon=0.0,
        log_stdout=True,
    ):
        pass
