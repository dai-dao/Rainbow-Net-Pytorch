import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.autograd import Variable

import numpy as np 
from baselines.common.schedules import LinearSchedule
from baselines import logger

from model import NoisyDistDuelingConv
from replay_buffer import ReplayBuffer_NStep, PrioritizedReplayBuffer_NStep


class RainbowAgent(object):
    def __init__(self, ob_shape, num_action, args):
        self.args = args
        self.num_action = num_action
        self.ob_chan, self.ob_w, self.ob_h = ob_shape

        self.dtype = torch.FloatTensor
        self.atype = torch.LongTensor
        if self.args.cuda:
            self.dtype = torch.cuda.FloatTensor
            self.atype = torch.cuda.LongTensor

        self.v_min = args.dist_params['Vmin']
        self.v_max = args.dist_params['Vmax']
        self.nb_atoms = args.dist_params['nb_atoms']
        self.dz = (self.v_max - self.v_min) / (self.nb_atoms - 1)
        self.z = torch.linspace(self.v_min, self.v_max, self.nb_atoms).type(self.dtype)
        self.m = torch.zeros(args.batch_size, self.nb_atoms).type(self.dtype)
        
        self.model = NoisyDistDuelingConv(self.nb_atoms, ob_shape[0], num_action, self.dtype, args.sigma_init)
        self.target_model = NoisyDistDuelingConv(self.nb_atoms, ob_shape[0], num_action, self.dtype, args.sigma_init)
        if self.args.cuda:
            self.model.cuda()
            self.target_model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, eps=self.args.adam_eps)

        # Helps in computing the update
        self.offset = torch.linspace(0, (args.batch_size - 1) * self.nb_atoms, args.batch_size).type(self.atype)
        self.offset = self.offset.unsqueeze(1).expand(args.batch_size, self.nb_atoms)


    def sample_noise(self):
        self.model.sample_noise()
        self.target_model.sample_noise()


    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    
    def p_to_q(self, phi):
        z_support = Variable(self.z.expand(phi.size(0), -1)).unsqueeze(2)
        q_out = torch.bmm(phi, z_support).view(-1, self.num_action)
        return q_out


    def act(self, ob):
        ob_var = Variable(torch.from_numpy(ob).contiguous().type(self.dtype)).view(-1, self.ob_chan, self.ob_w, self.ob_h)
        phi = self.model(ob_var)
        q_out = self.p_to_q(phi)
        _, deterministic_actions = q_out.data.max(1)
        out = deterministic_actions.cpu().numpy().astype(np.int32).reshape(-1)
        return out[0]


    def update(self, obs, actions, rewards, obs_next, dones, weights):
        """
            Reshape arrays to [n_step, batch_size, ..]
        """
        batch_size = self.args.batch_size
        batch_dim = self.atype(np.arange(batch_size))    
        gamma = self.args.gamma
        nb_atoms = self.nb_atoms 
        nstep = self.args.nstep

        obs = Variable(torch.from_numpy(obs).type(self.dtype)).view(nstep, batch_size, self.ob_chan, self.ob_w, self.ob_h)
        obs_next = Variable(torch.from_numpy(obs_next).type(self.dtype)).view(nstep, batch_size, self.ob_chan, self.ob_w, self.ob_h)
        weights = Variable(torch.from_numpy(weights).type(self.dtype)).view(batch_size, 1)
        actions = Variable(torch.from_numpy(actions.astype(int)).type(self.atype)).view(nstep, batch_size, 1)
        rewards = torch.from_numpy(rewards).type(self.dtype).view(nstep, batch_size, 1)
        dones = torch.from_numpy(dones.astype(float)).type(self.dtype).view(nstep, batch_size, 1) 

        # Last State of the nstep sequence to boostrap off of
        obs_last = obs_next[-1].view(batch_size, self.ob_chan, self.ob_w, self.ob_h)

        # DDQN
        next_online_phi = self.model(obs_last)
        next_online_q = self.p_to_q(next_online_phi)
        _, next_online_action = next_online_q.data.max(1)
        next_target_phi = self.target_model(obs_last).data
        next_target_best = next_target_phi[batch_dim, next_online_action]

        # Compute n-step q loss
        kl_losses = torch.zeros(batch_size).type(self.dtype)
        Tzj = torch.zeros(batch_size, nb_atoms).type(self.dtype)
        big_z = self.z.expand(batch_size, -1)

        for i in reversed(range(nstep)):
            m = self.m.fill_(0)
            rewards_i = rewards[i].clone().view(batch_size, 1)
            dones_i = dones[i].clone().view(batch_size, 1)
            actions_i = actions[i].clone().view(batch_size, 1)
            obs_i = obs[i].clone().view(batch_size, self.ob_chan, self.ob_w, self.ob_h)

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


def learn(env, args):
    logger.configure('./rainbow_log', ['stdout', 'csv'])

    ob = env.reset()
    ob_shape = ob.shape
    num_action = int(env.action_space.n)

    agent = RainbowAgent(ob_shape, num_action, args)
    replay_buffer = PrioritizedReplayBuffer_NStep(args.buffer_size, alpha=args.prioritized_replay_alpha)
    args.prioritized_replay_beta_iters = args.max_timesteps
    beta_schedule = LinearSchedule(args.prioritized_replay_beta_iters, 
                                    initial_p=args.prioritized_replay_beta0, 
                                    final_p=1.0)
    episode_rewards = [0.0]
    saved_mean_reward = None
    n_step_seq = []

    agent.sample_noise()
    agent.update_target()

    for t in range(args.max_timesteps):
        action = agent.act(ob)
        new_ob, rew, done, _ = env.step(action)
        # Append new step
        n_step_seq.append((ob, action, rew, new_ob, done))
        ob = new_ob

        episode_rewards[-1] += rew
        if done or t % args.max_steps_per_episode == 0:
            ob = env.reset()
            episode_rewards.append(0.0)

        # Add to experience replay once collect enough steps
        if len(n_step_seq) >= args.nstep:
           replay_buffer.add(n_step_seq)
           n_step_seq = []

        if t > args.learning_starts and t % args.replay_period == 0:
            # Replay
            experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(t))
            (obs_n, actions_n, rewards_n, obs_next_n, dones_n, weights, batch_idxes) = experience
            # Update network
            kl_errors = agent.update(obs_n, actions_n, rewards_n, obs_next_n, dones_n, weights)
            agent.sample_noise()
            # Update priorities in buffer
            replay_buffer.update_priorities(batch_idxes, kl_errors)

        if t > args.learning_starts and t % args.target_network_update_freq == 0:
            # Update target periodically
            agent.update_target()  

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and args.print_freq is not None and len(episode_rewards) % args.print_freq == 0:
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            logger.dump_tabular()