import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.autograd import Variable

import numpy as np 
from baselines.common.schedules import LinearSchedule

from model import DistMLP, DistDuelingMLP, NoisyDistDuelingMLP
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer_NStep, PrioritizedReplayBuffer_NStep


class DistDeepQ(object):
    def __init__(self, env, args):
        self.env = env
        self.args = args

        self.ob_space = int(np.prod(env.observation_space.shape))
        self.ac_space = int(env.action_space.n)

        self.dtype = torch.FloatTensor
        self.atype = torch.LongTensor
        if self.args.cuda:
            self.dtype = torch.cuda.FloatTensor
            self.atype = torch.cuda.LongTensor
            self.model.cuda()
            self.target_model.cuda()

        self.v_min = args.dist_params['Vmin']
        self.v_max = args.dist_params['Vmax']
        self.nb_atoms = args.dist_params['nb_atoms']
        self.dz = (self.v_max - self.v_min) / (self.nb_atoms - 1)
        self.z = torch.linspace(self.v_min, self.v_max, self.nb_atoms).type(self.dtype)
        self.m = torch.zeros(args.batch_size, self.nb_atoms).type(self.dtype)
        
        # self.model = DistDuelingMLP(64, self.nb_atoms, self.ob_space, self.ac_space)
        # self.target_model = DistDuelingMLP(64, self.nb_atoms, self.ob_space, self.ac_space)

        self.model = NoisyDistDuelingMLP(64, self.nb_atoms, self.ob_space, self.ac_space)
        self.target_model = NoisyDistDuelingMLP(64, self.nb_atoms, self.ob_space, self.ac_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        # Helps in update
        self.offset = torch.linspace(0, (args.batch_size - 1) * self.nb_atoms, args.batch_size).type(self.atype)
        self.offset = self.offset.unsqueeze(1).expand(args.batch_size, self.nb_atoms)


    def sample_noise(self):
        self.model.sample_noise()
        self.target_model.sample_noise()


    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    
    def p_to_q(self, phi):
        z_support = Variable(self.z.expand(phi.size(0), -1)).unsqueeze(2)
        q_out = torch.bmm(phi, z_support).view(-1, self.ac_space)
        return q_out


    def act_noisy_distributional(self, ob):
        ob_var = Variable(torch.from_numpy(ob).type(self.dtype)).view(-1, self.ob_space)
        phi = self.model(ob_var)
        q_out = self.p_to_q(phi)
        _, deterministic_actions = q_out.data.max(1)
        out = deterministic_actions.cpu().numpy().astype(np.int32).reshape(-1)
        return out[0]


    def act_distributional(self, ob, eps):
        ob_var = Variable(torch.from_numpy(ob).type(self.dtype)).view(-1, self.ob_space)
        phi = self.model(ob_var)
        q_out = self.p_to_q(phi)
        _, deterministic_actions = q_out.data.max(1)

        # Epsilon greedy actions
        batch_size = ob_var.size(0)
        random_actions = self.atype(batch_size).random_(0, self.ac_space)
        choose_random = self.dtype(batch_size).uniform_(0, 1) < eps
        deterministic_actions[choose_random == 1] = 0 
        random_actions[choose_random == 0] = 0
        stochastic_actions = deterministic_actions + random_actions
        out = stochastic_actions.cpu().numpy().astype(np.int32).reshape(-1)
        return out[0] 
    

    def distributional_nstep_update(self, obs, actions, rewards, obs_next, dones, weights):
        """
            Reshape arrays to [n_step, batch_size, ..]
        """
        batch_size = self.args.batch_size
        batch_dim = self.atype(np.arange(batch_size))    
        gamma = self.args.gamma
        nb_atoms = self.nb_atoms 
        nstep = self.args.nstep

        obs = Variable(torch.from_numpy(obs).type(self.dtype)).view(nstep, batch_size, self.ob_space)
        obs_next = Variable(torch.from_numpy(obs_next).type(self.dtype)).view(nstep, batch_size, self.ob_space)
        weights = Variable(torch.from_numpy(weights).type(self.dtype)).view(batch_size, 1)
        actions = Variable(torch.from_numpy(actions.astype(int)).type(self.atype)).view(nstep, batch_size, 1)
        rewards = torch.from_numpy(rewards).type(self.dtype).view(nstep, batch_size, 1)
        dones = torch.from_numpy(dones.astype(float)).type(self.dtype).view(nstep, batch_size, 1) 

        # Last State of the nstep sequence to boostrap off of
        obs_last = obs_next[-1].view(batch_size, self.ob_space)

        # DDQN
        next_online_phi = self.model(obs_last)
        next_online_q = self.p_to_q(next_online_phi)
        _, next_online_action = next_online_q.data.max(1)
        next_target_phi = self.target_model(obs_next).data
        next_target_best = next_target_phi[batch_dim, next_online_action]

        # Compute n-step q loss
        kl_losses = torch.zeros(batch_size).type(self.dtype)
        Tzj = torch.zeros(batch_size, nb_atoms)
        big_z = self.z.expand(batch_size, -1)

        for i in reversed(range(nstep)):
            m = self.m.fill_(0)
            rewards_i = rewards[i].view(batch_size, 1)
            dones_i = dones[i].view(batch_size, 1)
            actions_i = actions[i].view(batch_size, 1)
            obs_i = obs[i].view(batch_size, self.ob_space)

            big_r_i = rewards_i.expand(batch_size, nb_atoms)
            big_dones_i = dones_i.expand(batch_size, nb_atoms)
            Tzj.add_(big_r_i + gamma * (1.0 - big_dones_i) * big_z)
            bellman_i = torch.clamp(Tzj, self.v_min, self.v_max)

            b = (bellman_i - self.v_min) / self.dz
            l = b.floor().long()
            u = b.ceil().long()            

            m.view(-1).index_add_(0, (l + self.offset).view(-1),
                                (next_target_best * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + self.offset).view(-1),
                                (next_target_best * (b - l.float())).view(-1))
            online_phi_i = self.model(obs_i)
            q_out_selected_i = online_phi_i[batch_dim, actions_i.squeeze()]
            loss_i = -(Variable(m) * torch.log(q_out_selected_i + 1e-5)).sum(-1)
            kl_losses.add_(loss_i.data)

            self.optimizer.zero_grad()
            weighted_loss_i = (loss_i * weights).mean()
            weighted_loss_i.backward()
            nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_norm_clipping)
            self.optimizer.step() 

        return kl_losses.cpu().numpy().flatten()


    def distributional_update(self, obs, actions, rewards, obs_next, dones, weights):
        batch_size = self.args.batch_size
        batch_dim = self.atype(np.arange(batch_size))    
        gamma = self.args.gamma
        nb_atoms = self.nb_atoms

        obs = Variable(torch.from_numpy(obs).type(self.dtype)).view(-1, self.ob_space)
        obs_next = Variable(torch.from_numpy(obs_next).type(self.dtype)).view(-1, self.ob_space)
        weights = Variable(torch.from_numpy(weights).type(self.dtype)).view(-1, 1)
        actions = Variable(torch.from_numpy(actions.astype(int)).type(self.atype)).view(-1, 1)
        rewards = torch.from_numpy(rewards).type(self.dtype).view(-1, 1)
        dones = torch.from_numpy(dones.astype(float)).type(self.dtype).view(-1, 1)         

        # DDQN
        next_online_phi = self.model(obs_next)
        next_online_q = self.p_to_q(next_online_phi)
        _, next_online_action = next_online_q.data.max(1)
        next_target_phi = self.target_model(obs_next).data
        next_target_best = next_target_phi[batch_dim, next_online_action]

        big_r = rewards.expand(batch_size, self.nb_atoms)
        big_dones = dones.expand(batch_size, self.nb_atoms)
        big_z = self.z.expand(batch_size, -1)
        
        # Compute projection of Bellman operator
        bellman_op = torch.clamp(big_r + gamma * big_z * (1.0 - big_dones), self.v_min, self.v_max)

        # Compute categorical indices for distributing the probabilities
        b = (bellman_op - self.v_min) / self.dz
        l = b.floor().long()
        u = b.ceil().long()

        # Distribute probabilities
        m = self.m.fill_(0)
        m.view(-1).index_add_(0, (l + self.offset).view(-1),
                                (next_target_best * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + self.offset).view(-1),
                                (next_target_best * (b - l.float())).view(-1))

        online_phi = self.model(obs)
        q_out_selected = online_phi[batch_dim, actions.squeeze()]
        cross_entropy_losses = -(Variable(m) * torch.log(q_out_selected + 1e-5)).sum(-1)
        weighted_loss = (cross_entropy_losses * weights).mean()
        kl_errors = cross_entropy_losses.data.cpu().numpy().flatten()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_norm_clipping)
        self.optimizer.step() 
        return kl_errors



def learn_nstep(env, args, callback=None):
    dist_deepq = DistDeepQ(env, args)

    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer_NStep(args.buffer_size, alpha=args.prioritized_replay_alpha)
        args.prioritized_replay_beta_iters = args.max_timesteps
        beta_schedule = LinearSchedule(args.prioritized_replay_beta_iters, 
                                             initial_p=args.prioritized_replay_beta0, 
                                             final_p=1.0)
    else:
        replay_buffer = ReplayBuffer_NStep(args.buffer_size)
        beta_schedule = None
    
    exploration = LinearSchedule(schedule_timesteps=int(args.exploration_fraction * args.max_timesteps),
                                 initial_p=1.0,
                                 final_p=args.exploration_final_eps)

    dist_deepq.sample_noise()
    dist_deepq.update_target()
    episode_rewards = [0.0]
    saved_mean_reward = None
    ob = env.reset()
    n_step_seq = []

    for t in range(args.max_timesteps):
        if callback is not None:
            if callback(locals(), globals()):
                break

        update_eps = exploration.value(t)
        # action = dist_deepq.act_distributional(ob, update_eps)
        action = dist_deepq.act_noisy_distributional(ob)
        new_ob, rew, done, _ = env.step(action)
        # Append new step
        n_step_seq.append((ob, action, rew, new_ob, done))
        ob = new_ob

        episode_rewards[-1] += rew
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
            reset = True

        # Add to experience replay once collect enough steps
        if len(n_step_seq) >= args.nstep:
           replay_buffer.add(n_step_seq)
           n_step_seq = []

        if t > args.learning_starts and t % args.replay_period == 0:
            if args.prioritized_replay:
                experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(t))
                (obs_n, actions_n, rewards_n, obs_next_n, dones_n, weights, batch_idxes) = experience
            else:
                obs_n, actions_n, rewards_n, obs_next_n, dones_n = replay_buffer.sample(args.batch_size)
                weights, batch_idxes = np.ones(args.batch_size), None

            kl_errors = dist_deepq.distributional_nstep_update(obs_n, actions_n, rewards_n, obs_next_n, dones_n, weights)
            dist_deepq.sample_noise()

            if args.prioritized_replay:
                replay_buffer.update_priorities(batch_idxes, kl_errors)

        if t > args.learning_starts and t % args.target_network_update_freq == 0:
            dist_deepq.update_target()  

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and args.print_freq is not None and len(episode_rewards) % args.print_freq == 0:
            print('steps {} episodes {} mean reward {}'.format(t, num_episodes, mean_100ep_reward))



def learn(env, args, callback=None): 
    dist_deepq = DistDeepQ(env, args)

    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(args.buffer_size, alpha=args.prioritized_replay_alpha)
        args.prioritized_replay_beta_iters = args.max_timesteps
        beta_schedule = LinearSchedule(args.prioritized_replay_beta_iters, 
                                             initial_p=args.prioritized_replay_beta0, 
                                             final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(args.buffer_size)
        beta_schedule = None
    
    exploration = LinearSchedule(schedule_timesteps=int(args.exploration_fraction * args.max_timesteps),
                                 initial_p=1.0,
                                 final_p=args.exploration_final_eps)

    # dist_deepq.sample_noise()
    dist_deepq.update_target()
    episode_rewards = [0.0]
    saved_mean_reward = None
    ob = env.reset()

    for t in range(args.max_timesteps):
        if callback is not None:
            if callback(locals(), globals()):
                break

        update_eps = exploration.value(t)
        action = dist_deepq.act_distributional(ob, update_eps)
        # action = dist_deepq.act_noisy_distributional(ob)
        new_ob, rew, done, _ = env.step(action)
        replay_buffer.add(ob, action, rew, new_ob, float(done))
        ob = new_ob

        episode_rewards[-1] += rew
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
            reset = True

        if t > args.learning_starts and t % args.train_freq == 0:
            if args.prioritized_replay:
                experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(t))
                (obs, actions, rewards, obs_next, dones, weights, batch_idxes) = experience
            else:
                obs, actions, rewards, obs_next, dones = replay_buffer.sample(args.batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            
            kl_errors = dist_deepq.distributional_update(obs, actions, rewards, obs_next, dones, weights)
            # dist_deepq.sample_noise()

            if args.prioritized_replay:
                replay_buffer.update_priorities(batch_idxes, kl_errors)

        if t > args.learning_starts and t % args.target_network_update_freq == 0:
            dist_deepq.update_target()  

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and args.print_freq is not None and len(episode_rewards) % args.print_freq == 0:
            print('steps {} episodes {} mean reward {}'.format(t, num_episodes, mean_100ep_reward))
            '''
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            logger.dump_tabular()
            '''