import torch


class CartPoleParams:
    def __init__(self):
        self.cuda = torch.cuda.is_available()

        self.lr = 1e-3
        self.max_timesteps = 100000
        self.buffer_size = 50000
        self.exploration_fraction = 0.1
        self.exploration_final_eps = 0.02
        self.print_freq = 10
        self.env_id = "CartPole-v0"

        self.train_freq = 1
        self.batch_size = 32
        self.checkpoint_freq = 10000
        self.learning_starts = 1000
        self.gamma = 0.99
        self.target_network_update_freq = 500
        self.prioritized_replay = True
        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay_beta0 = 0.4
        self.prioritized_replay_beta_iters = None
        self.prioritized_replay_eps = 1e-6
        self.grad_norm_clipping = 10
        self.param_noise = False

        # Rainbow param (DeepMind papers)
        self.n_step = 3
        self.replay_period = 1
        self.frame_skip = 4
        self.training_starts = 80000 / 4
        self.target_update_freq = 32000 / 4
        self.frame_stack = 4
        self.memory_size = 10e6
        self.reward_clipping = True # [-1, 1]