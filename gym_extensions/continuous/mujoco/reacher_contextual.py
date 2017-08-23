PYTHONPATH = '~/Documents/gym-extensions/'
import sys
sys.path.append(PYTHONPATH)
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os.path as osp

from gym.envs.mujoco.reacher import ReacherEnv
try:
    import mujoco_py
    from mujoco_py.mjlib import mjlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

import os
import gym
import gym_extensions


class ReacherContextualEnv(ReacherEnv):

    def __init__(self, *args, **kwargs):
        ReacherEnv.__init__(self)
        # the context is a 4-dim vector [x1, y1, x2, y2]
        # (x1,y1) - coords of the tip of reacher; (x2,y2) - coords of the target
        self.context   = np.array([0.1, 0.1, 0.1, 0.1])
        self.policy_type = ""
        self.context_high = np.array([  i*2 for i in self.context])
        self.context_low  = np.array([ -i*2 for i in self.context]) # the params in the context can't be less or equal to zero!
        self.bias = 0
        self.weights = [0]*self.observation_space.shape[0]
        
    def _step(self, action):
        state, reward, done, _  = super(ReacherContextualEnv, self)._step(action)
        return state, reward, done, {}

    def change_context(self, context_vector):
        # the context is a 4-dim vector [x1, y1, x2, y2]
        # (x1,y1) - coords of the tip of reacher; (x2,y2) - coords of the target

        qpos  = np.array(context_vector)
        qvel = self.init_qvel
        qvel[-2:] = 0
        self.set_state(qpos, qvel)

    def set_policy_type(self, policy_type):
        self.policy_type = policy_type

    def context_space_info(self):
        context_info_dict = {}
        context_info_dict['context_vals'] = self.context
        context_info_dict['context_dims'] = len(self.context)
        context_info_dict['context_high'] = self.context_high.tolist()
        context_info_dict['context_low' ] = self.context_low.tolist()
        context_info_dict['state_dims'  ] = self.observation_space.shape[0]
        # I need to know what the size of the action vector I need to pass to the transition function
        context_info_dict['action_dims' ] = self.action_space.shape[0]
        context_info_dict['action_space'] = 'continuous'
        context_info_dict['state_high'  ] = self.observation_space.high.tolist()
        context_info_dict['state_low'   ] = self.observation_space.low.tolist()
        context_info_dict['action_high' ] = self.action_space.high.tolist()
        context_info_dict['action_low'  ] = self.action_space.low.tolist()

        return context_info_dict


if __name__ == "__main__":
    import time
    #env = gym.make('Reacher-v1')
    env = gym.make('ReacherContextual-v0')
    for i_episode in range(500):
        env.reset()
        while True:
            goal = np.random.uniform(low=-.25, high=.25, size=4)
            if np.linalg.norm(goal) < 2:
                break
        env.unwrapped.change_context(goal)
        print 'target', env.unwrapped.get_body_com("target")
        print 'qpos', env.unwrapped.model.data.qpos
        time.sleep(2)
        #print env.unwrapped.context_space_info()
        #print env.unwrapped.weights
        print env.unwrapped.model.nq
        for t in range(500):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            #print observation
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break