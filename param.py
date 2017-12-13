import torch


class Params():
    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.lr = 3e-4
        self.max_timesteps = 100000
        self.buffer_size = 50000
        self.exploration_fraction = 0.1
        self.exploration_final_eps = 0.02
        self.print_freq = 10
        self.target_network_update_freq = 500
        self.batch_size = 32
        self.gamma = 0.95
        # Sensitive param, use [-10, 10, 51] for Atari for sure
        self.dist_params = {'Vmin': 0, 'Vmax': 25, 'nb_atoms': 11}
        self.train_freq = 1
        self.checkpoint_freq = 10000
        self.learning_starts = 1000
        self.prioritized_replay = True
        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay_beta0 = 0.4
        self.prioritized_replay_beta_iters = None
        self.prioritized_replay_eps = 1e-6
        self.num_cpu = 16
        self.param_noise = False
        self.grad_norm_clipping = 10
        
        # Not good for CartPole
        self.nstep = 3
        self.replay_period = 1
