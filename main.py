
import gym
import torch 
import os 
import numpy as np 

from params import *
from deepq import learn, learn_n_step


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def train_cartpole():
    args = CartPoleParams()
    env = gym.make(args.env_id)
    # learn(env, args=args)
    learn_n_step(env, args=args)


def main():
    train_cartpole()


if __name__ == '__main__':
    main()