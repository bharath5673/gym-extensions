import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os.path as osp

from gym.envs.mujoco.hopper import HopperEnv
try:
    import mujoco_py
    from mujoco_py.mjlib import mjlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

import os
import gym
import gym_extensions


class HopperContextualEnv(HopperEnv):

    def __init__(self, *args, **kwargs):
        HopperEnv.__init__(self)
        self.context   = np.copy(self.model.body_mass)[1:]
        self.policy_type = ""
        self.context_high = np.array([i[0]*10  for i in self.context])
        self.context_low  = np.array([i[0]*0.1 for i in self.context]) # the params in the context can't be less or equal to zero!
        self.bias = 0
        self.weights = [0]*self.observation_space.shape[0]
        
    def _step(self, action):
        state, reward, done, _  = super(HopperContextualEnv, self)._step(action)
        return state, reward, done, {}

    def change_context(self, context_vector):
        """The body names of the Hopper are ['world', 'torso', 'thigh', 'leg', 'foot']
           obtained as: env.unwrapped.model.body_names
        """
        temp = np.copy(self.model.body_mass)
        temp[1:] = [[i]for i in context_vector]
        #temp[1] = context_vector
        self.model.body_mass = temp
        self.model._compute_subtree()  # pylint: disable=W0212
        self.model.forward()

    def set_policy_type(self, policy_type):
        self.policy_type = policy_type

    def context_space_info(self):
        context_info_dict = {}
        context_info_dict['context_vals'] = self.context
        context_info_dict['context_dims'] = len(self.context)
        context_info_dict['context_vals'] = [i[0] for i in self.context]
        context_info_dict['context_high'] = self.context_high.tolist()
        context_info_dict['context_low' ] = self.context_low.tolist()
        context_info_dict['state_dims'  ] = self.observation_space.shape[0]
        # I need to know what the size of the action vector I need to pass to the transition function
        context_info_dict['action_dims' ] = self.action_space.shape[0]
        context_info_dict['action_space'] = 'continuous'
        context_info_dict['state_high'  ] = [1000 for i in self.observation_space.high.tolist()]
        context_info_dict['state_low'   ] = [-1000 for i in self.observation_space.low.tolist()]
        context_info_dict['action_high' ] = self.action_space.high.tolist()
        context_info_dict['action_low'  ] = self.action_space.low.tolist()

        return context_info_dict


if __name__ == "__main__":
    import time
    env = gym.make('HopperContextual-v0')
    for i_episode in range(500):
        env.reset()
        env.unwrapped.change_context([0.1,0.1,0.1,0.1])
        print 'body_mass ', env.unwrapped.model.body_mass
        time.sleep(2)
        print env.unwrapped.context_space_info()
        print env.unwrapped.weights
        for t in range(500):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
