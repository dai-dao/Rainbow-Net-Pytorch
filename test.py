import gym
import torch 
from torch.autograd import Variable
import os 
import numpy as np 

from params import *
from deepq import *
from models import *
import utils


class TestSuite(object):
    def __init__(self):
        pass


    def test_net_construct_plus(self):
        net = MLP_PLUS(4, 2, [64, 64])
        print('Parameters are')
        for param in net.parameters():
            print(param.size())


    def test_net_construct_failed(self):
        net = MLP_FAILED(4, 2, [64])
        print('Parameters are')
        for param in net.parameters():
            print(param.size())


    def test_act(self):
        args = CartPoleParams()
        env = gym.make(args.env_id)

        q_func = MLP
        deepq = DeepQ(env, q_func, args)
        exploration = utils.LinearSchedule(schedule_timesteps=int(args.exploration_fraction * args.max_timesteps),
                                            initial_p=1.0,
                                            final_p=args.exploration_final_eps)

        t = 1
        eps = exploration.value(t)
    
        obs = env.reset()
        action = deepq.act(obs, eps)
        print('Output action', action)


    def test_stochastic_act(self):
        deterministic_actions = torch.LongTensor([0, 1, 2, 3, 4, 5, 5, 5, 5, 5])
        print('Deterministic actions', deterministic_actions.numpy())

        random_actions = torch.LongTensor(10).random_(0, 5)
        print('Random actions', random_actions.numpy())

        # 90% chance to choose random action
        choose_random = torch.FloatTensor(10).uniform_(0, 1) < 0.4
        print('Choose random', choose_random.numpy())

        deterministic_actions[choose_random == 1] = 0 
        random_actions[choose_random == 0] = 0
        stochastic_actions = deterministic_actions + random_actions
        print('Stochastic actions', stochastic_actions.numpy())


    def test_net(self, env, net):
        obs = env.reset()
        obs = np.reshape(obs, (-1, ob_space))

        test_tensor = Variable(torch.from_numpy(obs).float())
        test_output = net(test_tensor)
        print('Output tensor size', test_output.size())


if __name__ == "__main__":
    test_suite = TestSuite()
    # test_suite.test_act()
    test_suite.test_net_construct_plus()