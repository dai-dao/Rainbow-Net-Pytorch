import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utils
from models import *
from replay import *

from collections import deque
from baselines import logger
import numpy as np 
import random 


class DeepQ(object):
    def __init__(self, env, args):
        self.env = env
        self.args = args

        self.ob_space = int(np.prod(env.observation_space.shape))
        self.ac_space = env.action_space.n

        self.net = MLP_Dueling(self.ob_space, self.ac_space, [64])    
        self.target_net = MLP_Dueling(self.ob_space, self.ac_space, [64])

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss(reduce=False)


    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())           


    def act(self, ob_var, eps, stochastic=True):
        ob_var = ob_var.view(-1, self.ob_space)
        q_out = self.net(ob_var)
        _, deterministic_actions = q_out.max(1)
        deterministic_actions = deterministic_actions.data

        batch_size = ob_var.size(0)
        random_actions = torch.LongTensor(batch_size).random_(0, self.ac_space)
        choose_random = torch.Tensor(batch_size).uniform_(0, 1) < eps
        deterministic_actions[choose_random == 1] = 0 
        random_actions[choose_random == 0] = 0
        stochastic_actions = deterministic_actions + random_actions
        out = stochastic_actions.cpu().numpy().astype(np.int32).reshape(-1)
        return out[0] 

    
    def NStep_update(self, obs_n, actions_n, rewards_n, obs_next_n, dones_n, weights):
        '''
            Reshape to [n_step, batch_size, ..]
        '''
        obs = Variable(torch.from_numpy(obs_n).type(torch.FloatTensor)).view(self.args.n_step, self.args.batch_size, self.ob_space)
        next_obs = Variable(torch.from_numpy(obs_next_n).type(torch.FloatTensor)).view(self.args.n_step, self.args.batch_size, self.ob_space)
        actions = Variable(torch.from_numpy(actions_n.astype(int)).type(torch.LongTensor)).view(self.args.n_step, self.args.batch_size, 1)
        weights = Variable(torch.from_numpy(weights).type(torch.FloatTensor)).view(self.args.batch_size, 1)
        rewards = torch.from_numpy(rewards_n).type(torch.FloatTensor).view(self.args.n_step, self.args.batch_size, 1)
        dones = torch.from_numpy(dones_n.astype(float)).type(torch.FloatTensor).view(self.args.n_step, self.args.batch_size, 1)
        
        # Last State of the nstep sequence
        last_ob = next_obs[-1].view(self.args.batch_size, self.ob_space)
        
        # DQN
        # last_target, _ = self.target_net(last_ob).data.max(1) 
        
        # DDQN
        last_target_value = self.target_net(last_ob).data
        _, last_online_action = self.net(last_ob).data.max(1)
        last_target = last_target_value.gather(1, last_online_action.view(-1, 1))

        # Computing n_step q loss
        td_n_step_errors = torch.zeros(self.args.batch_size, 1)
        losses = Variable(torch.zeros(self.args.batch_size, 1))

        for i in reversed(range(self.args.n_step)):
            # In-place ops, since we're propagating this boostrapped value down the sequence
            last_target.mul_(self.args.gamma).mul_(1.0 - dones[i]).add_(rewards[i])
            online_q = self.net(obs[i])
            online_q_out = online_q.gather(1, actions[i])
            td_n_step_errors.add_(online_q_out.data - last_target)
            losses.add_(self.huber_loss(online_q_out, Variable(last_target)))
        weighted_loss = (weights * losses).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm(self.net.parameters(), self.args.grad_norm_clipping)
        self.optimizer.step()
        return td_n_step_errors.cpu().numpy().flatten()


    def update(self, obs, actions, rewards, next_obs, dones, weights):
        obs = Variable(torch.from_numpy(obs).type(torch.FloatTensor)).view(-1, self.ob_space)
        next_obs = Variable(torch.from_numpy(next_obs).type(torch.FloatTensor)).view(-1, self.ob_space)
        rewards = Variable(torch.from_numpy(rewards).type(torch.FloatTensor)).view(-1, 1)
        weights = Variable(torch.from_numpy(weights).type(torch.FloatTensor)).view(-1, 1)
        dones = Variable(torch.from_numpy(dones.astype(float)).type(torch.FloatTensor)).view(-1, 1)
        actions = Variable(torch.from_numpy(actions.astype(int)).type(torch.LongTensor)).view(-1, 1)
                
        q_out = self.net(obs)
        q_out_selected = q_out.gather(1, actions)

        # Double Q, compute the Q-value of the next state based on the target network
        # based on action chosen by the online network
        next_q_out_online = self.net(next_obs).detach()
        next_q_out_target = self.target_net(next_obs).detach()
        _, next_q_best_online = next_q_out_online.max(1)
        next_q_best = next_q_out_target.gather(1, next_q_best_online.view(-1, 1))

        # compute RHS of bellman equation
        # 0 Q-Value if episode terminates
        next_q_best_masked = (1.0 - dones) * next_q_best
        q_out_selected_target = rewards + self.args.gamma * next_q_best_masked

        # compute the error (potentially clipped)
        td_errors = (q_out_selected - q_out_selected_target).data.cpu().numpy().flatten()
        losses = self.huber_loss(q_out_selected, q_out_selected_target)
        weighted_loss = (losses * weights).sum()

        # loss = self.criterion(q_out_selected, q_out_selected_target)
        
        # Optimizer step
        self.optimizer.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm(self.net.parameters(), self.args.grad_norm_clipping)
        self.optimizer.step()
        return td_errors


def learn_n_step(env, args):
    deepq = DeepQ(env, args)
    exploration = utils.LinearSchedule(schedule_timesteps=int(args.exploration_fraction * args.max_timesteps),
                                        initial_p=1.0,
                                        final_p=args.exploration_final_eps)
    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer_NStep(args.buffer_size, args.n_step, alpha=args.prioritized_replay_alpha)
        args.prioritized_replay_beta_iters = args.max_timesteps
        beta_schedule = utils.LinearSchedule(args.prioritized_replay_beta_iters, 
                                             initial_p=args.prioritized_replay_beta0, 
                                             final_p=1.0)
    else:
        replay_buffer = ReplayBuffer_NStep(args.buffer_size, args.n_step)
        beta_schedule = None

    # Copy params to target network
    deepq.update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    ob = env.reset()
    reset = True
    n_step_seq = []

    for t in range(args.max_timesteps):
        update_eps = exploration.value(t)
        update_param_noise_threshold = 0.

        ob_var = Variable(torch.from_numpy(ob).type(torch.FloatTensor))
        action = deepq.act(ob_var, update_eps)
        reset = False
        new_ob, rew, done, _ = env.step(action)

        # Append new step 
        n_step_seq.append((ob, action, rew, new_ob, done))
        ob = new_ob

        episode_rewards[-1] += rew
        if done:
            ob = env.reset()
            episode_rewards.append(0.0)
            reset = True

        # Add to experience replay once collect enough steps
        if len(n_step_seq) >= args.n_step:
           replay_buffer.add(n_step_seq)
           n_step_seq = []

        if t > args.learning_starts and t % args.replay_period == 0:
            if args.prioritized_replay:
                experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(t))
                (obs_n, actions_n, rewards_n, obs_next_n, dones_n, weights, batch_idxes) = experience
            else:
                obs_n, actions_n, rewards_n, obs_next_n, dones_n = replay_buffer.sample(args.batch_size)
                weights, batch_idxes = np.ones(args.batch_size), None
            # N-step update
            td_errors = deepq.NStep_update(obs_n, actions_n, rewards_n, obs_next_n, dones_n, weights)

            # Update priority if using prioritized replay buffer
            if args.prioritized_replay:
                new_priorities = np.abs(td_errors) + args.prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

        if t > args.learning_starts and t % args.target_network_update_freq * args.replay_period == 0:
            # Update target network periodically.
            deepq.update_target()

        if done and args.print_freq is not None and len(episode_rewards) % args.print_freq == 0:
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            print('Step {} episode {} mean 100ep reward {}'.format(t, num_episodes, mean_100ep_reward))



def learn(env, args):
    # logger.configure('./log', ['csv', 'stdout'])
    deepq = DeepQ(env, args)
    exploration = utils.LinearSchedule(schedule_timesteps=int(args.exploration_fraction * args.max_timesteps),
                                        initial_p=1.0,
                                        final_p=args.exploration_final_eps)
    if args.prioritized_replay:
        replay_buffer = utils.PrioritizedReplayBuffer(args.buffer_size, alpha=args.prioritized_replay_alpha)
        args.prioritized_replay_beta_iters = args.max_timesteps
        beta_schedule = utils.LinearSchedule(args.prioritized_replay_beta_iters, 
                                             initial_p=args.prioritized_replay_beta0, 
                                             final_p=1.0)
    else:
        replay_buffer = utils.ReplayBuffer(args.buffer_size)
        beta_schedule = None

    # Copy params to target network
    deepq.update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    ob = env.reset()
    reset = True

    for t in range(args.max_timesteps):
        update_eps = exploration.value(t)
        update_param_noise_threshold = 0.

        ob_var = Variable(torch.from_numpy(ob).type(torch.FloatTensor))
        action = deepq.act(ob_var, update_eps)
        reset = False
        new_ob, rew, done, _ = env.step(action)
        # Store transition in the replay buffer.
        replay_buffer.add(ob, action, rew, new_ob, float(done))
        ob = new_ob

        episode_rewards[-1] += rew
        if done:
            ob = env.reset()
            episode_rewards.append(0.0)
            reset = True

        if t > args.learning_starts and t % args.train_freq == 0:
            if args.prioritized_replay:
                experience = r.eplay_buffer.sample(args.batch_size, beta=beta_schedule.value(t))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            # Do training
            td_errors = deepq.update(obses_t, actions, rewards, obses_tp1, dones, weights)

            # Update priority if using prioritized replay buffer
            if args.prioritized_replay:
                new_priorities = np.abs(td_errors) + args.prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

        if t > args.learning_starts and t % args.target_network_update_freq == 0:
            # Update target network periodically.
            deepq.update_target()

        if done and args.print_freq is not None and len(episode_rewards) % args.print_freq == 0:
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            print('Step {} episode {} mean 100ep reward {}'.format(t, num_episodes, mean_100ep_reward))


        '''
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and args.print_freq is not None and len(episode_rewards) % args.print_freq == 0:
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            logger.dump_tabular()

        if (args.checkpoint_freq is not None and t > args.learning_starts and
            num_episodes > 100 and t % args.checkpoint_freq == 0):
            if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                if args.print_freq is not None:
                    logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                saved_mean_reward, mean_100ep_reward))
                torch.save(deepq.net, args.env_id + '_net.pt')
                model_saved = True
                saved_mean_reward = mean_100ep_reward

        '''