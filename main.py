import gym
import numpy as np
import random
import torch

import dist_deepq
from model import DistMLP
from param import Params, AtariParams
import atari_rainbow

from baselines.common.atari_wrappers import make_atari, wrap_deepmind



def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def set_global_seeds(i, args):
    if args.cuda:
        torch.cuda.manual_seed(i)
    else:
        torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)


class wrap_pytorch(gym.ObservationWrapper):
    def _observation(self, observation):
        return np.array(observation).transpose(2, 0, 1)


def run_atari(env_id):
    args = AtariParams(env_id)
    env = make_atari(env_id)
    env = wrap_deepmind(env, clip_rewards=True, frame_stack=True, scale=True)
    env = wrap_pytorch(env)
    set_global_seeds(10, args)
    atari_rainbow.learn(env, args)


def run_cartpole():
    args = Params("CartPole-v0")
    env = gym.make("CartPole-v0")
    set_global_seeds(10, args)
    atari_rainbow.learn(env, args)


def main():
    env = gym.make("CartPole-v0")
    # env = gym.make("MountainCar-v0")
    env.seed(1337)
    # np.random.seed(1337)
    torch.manual_seed(1337)

    # dist_deepq.learn(env, args, callback=callback)
    # dist_deepq.learn_nstep(env, args, callback=callback)


if __name__ == '__main__':
    # main()
    # run_atari('BreakoutNoFrameskip-v4')
    run_cartpole()