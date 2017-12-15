import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np 
from baselines.common.schedules import LinearSchedule
from baselines import logger

from model import NoisyDistDuelingConv, NoisyDistDuelingMLP, NoisyDuelingConv, NoisyDuelingMLP
from replay_buffer import ReplayBuffer_NStep, PrioritizedReplayBuffer_NStep, PrioritizedReplayBuffer


class TestAgent(object):
    def __init__(self, ob_shape, num_action, args):
        self.args = args
        self.num_action = num_action
        self.ob_shape = ob_shape    

        self.dtype = torch.FloatTensor
        self.atype = torch.LongTensor
        if self.args.cuda:
            self.dtype = torch.cuda.FloatTensor
            self.atype = torch.cuda.LongTensor

        # self.model = NoisyDistDuelingConv(self.nb_atoms, ob_shape[0], num_action, self.dtype, args.sigma_init)
        # self.target_model = NoisyDistDuelingConv(self.nb_atoms, ob_shape[0], num_action, self.dtype, args.sigma_init)
        # self.model = NoisyDuelingConv(ob_shape[0], num_action, self.dtype, args.sigma_init)
        # self.target_model = NoisyDuelingConv(ob_shape[0], num_action, self.dtype, args.sigma_init)        
        self.model = NoisyDuelingMLP(ob_shape[0], num_action, self.dtype, 0.17)
        self.target_model = NoisyDuelingMLP(ob_shape[0], num_action, self.dtype, 0.17)

        if self.args.cuda:
            self.model.cuda()
            self.target_model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, eps=self.args.adam_eps)
        self.criterion = nn.MSELoss(reduce=False)
        self.huber_loss = nn.SmoothL1Loss(reduce=False)


    def sample_noise(self):
        self.model.sample_noise()
        self.target_model.sample_noise()


    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    
    def act(self, ob):
        ob_var = Variable(torch.from_numpy(ob).contiguous().type(self.dtype)).view(-1, *self.ob_shape)
        q_out = self.model(ob_var)
        _, deterministic_actions = q_out.data.max(1)
        out = deterministic_actions.cpu().numpy().astype(np.int32).reshape(-1)
        return out[0]


    def update(self, obs, actions, rewards, next_obs, dones, weights):
        obs = Variable(torch.from_numpy(obs).type(torch.FloatTensor)).view(-1, 4)
        next_obs = Variable(torch.from_numpy(next_obs).type(torch.FloatTensor)).view(-1, 4)
        dones = Variable(torch.from_numpy(dones.astype(float)).type(torch.FloatTensor)).view(-1, 1)
        rewards = Variable(torch.from_numpy(rewards).type(torch.FloatTensor)).view(-1, 1)
        actions = Variable(torch.from_numpy(actions.astype(int)).type(torch.LongTensor)).view(-1, 1)

        # Compute Bellman loss -> DDQN
        q_next = self.target_model(next_obs).detach()
        _, best_actions = self.model(next_obs).detach().max(1)
        q_next_best = q_next.gather(1, best_actions.view(-1, 1))
        q_next_best_rhs = rewards + self.args.gamma * q_next_best * (1 - dones)
        q = self.model(obs)
        q = q.gather(1, actions).squeeze(1)
        loss = self.criterion(q, q_next_best_rhs)
        
        # Step optimizer
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        
        return loss.data.numpy().flatten()


    def test_update(self, obs, actions, rewards, obs_next, dones, weights):
        obs = Variable(torch.from_numpy(obs).type(self.dtype)).view(-1, *self.ob_shape)
        obs_next = Variable(torch.from_numpy(obs_next).type(self.dtype)).view(-1, *self.ob_shape)
        weights = Variable(torch.from_numpy(weights).type(self.dtype)).view(-1, 1)
        actions = Variable(torch.from_numpy(actions.astype(int)).type(self.atype)).view(-1, 1)
        rewards = torch.from_numpy(rewards).type(self.dtype).view(-1, 1)
        dones = torch.from_numpy(dones.astype(float)).type(self.dtype).view(-1, 1)     

        # 
        online_q = self.model(obs)
        online_q_selected = online_q.gather(1, actions)

        # DDQN
        next_online_q = self.model(obs_next)
        _, next_online_action = next_online_q.data.max(1)
        next_target_q = self.target_model(obs_next).data

        next_target_best = next_target_q.gather(1, next_online_action.view(-1, 1))
        targets = rewards + (1.0 - dones) * self.args.gamma * next_target_best

        # Error 
        td_error = online_q_selected.data - targets 
        errors = F.smooth_l1_loss(online_q_selected, Variable(targets), reduce=False)
        weighted_error = (errors * weights).mean()

        # 
        self.optimizer.zero_grad()
        weighted_error.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_norm_clipping)
        self.optimizer.step()
        return td_error.cpu().numpy().flatten()


def learn(env, args):
    ob = env.reset()
    ob_shape = ob.shape
    num_action = int(env.action_space.n)

    agent = TestAgent(ob_shape, num_action, args)
    replay_buffer = PrioritizedReplayBuffer(args.buffer_size, alpha=args.prioritized_replay_alpha)
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
        replay_buffer.add(ob, action, rew, new_ob, float(done))
        ob = new_ob

        episode_rewards[-1] += rew
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
            reset = True

        if t > args.learning_starts and t % args.replay_period == 0:
            experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(t))
            (obs, actions, rewards, obs_next, dones, weights, batch_idxes) = experience
            agent.sample_noise()
            kl_errors = agent.update(obs, actions, rewards, obs_next, dones, weights)
            replay_buffer.update_priorities(batch_idxes, np.abs(kl_errors) + 1e-6)

        if t > args.learning_starts and t % args.target_network_update_freq == 0:
            agent.update_target()  

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and args.print_freq is not None and len(episode_rewards) % args.print_freq == 0:
            print('steps {} episodes {} mean reward {}'.format(t, num_episodes, mean_100ep_reward))