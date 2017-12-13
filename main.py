import gym
import numpy as np
import torch

import dist_deepq
from model import DistMLP
from param import Params


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    args = Params()    
    env = gym.make("CartPole-v0")
    # env = gym.make("MountainCar-v0")
    env.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)

    # dist_deepq.learn(env, args, callback=callback)
    dist_deepq.learn_nstep(env, args, callback=callback)


if __name__ == '__main__':
    main()
