PYTHONPATH = '~/Documents/gym-extensions/'
import sys
sys.path.append(PYTHONPATH)
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
    
    def change_context(self, i):
        #self.model.data.qpos = self.init_qpos + self.np_random.uniform(low=-.001, high=.001, size=self.model.nq)
        #self.model.data.qvel = self.init_qvel + self.np_random.uniform(low=-.001, high=.001, size=self.model.nv)
        #(mujoco_py.mjtypes.c_double * 3)(*[0., 0., gravity])
        temp = np.copy(self.model.body_mass)
        temp[1] *= float(i)
        self.model.body_mass = temp
        self.model._compute_subtree()  # pylint: disable=W0212
        self.model.forward()

    #self.model.body_mass = self.get_and_modify_bodymass(body_part, mass_scale)
    #        self.model._compute_subtree()
    #        self.model.forward()

    #def get_and_modify_bodymass(self, body_name, scale):
    #    idx = self.model.body_names.index(six.b(body_name))
    #    temp = np.copy(self.model.body_mass)
    #    temp[idx] *= scale
    #    return temp

if __name__ == "__main__":
    import time
    ###env = gym.make('HopperContextual-v0')
    env = gym.make('Striker-v0')
    #time_steps = 0
    #start_time = time.time()
    for i_episode in range(500):
        #observation = env.reset()
        ###print 'body_mass ', env.unwrapped.model.body_mass
        ###print 'waiting for the mass to change...'
        ###time.sleep(2)
        env.reset()
        ###env.unwrapped.change_context(i_episode+1)
        #print env.unwrapped.model.data.qpos
        #print env.unwrapped.model.data.qvel
        #print env.unwrapped.model.data.qfrc_applied
        ###print env.unwrapped.model.body_names
        #print env.unwrapped.model.body_names[1]
        #print 'com_subtree[1] ', env.unwrapped.model.data.com_subtree[1]
        #print 'nbody ', env.unwrapped.model.nbody
        #print 'body_ipos ', env.unwrapped.model.body_ipos
        ###print 'body_mass ', env.unwrapped.model.body_mass
        ###print 'now the mass should be ', i_episode+1
        ###time.sleep(2)
        # (mujoco_py.mjtypes.c_double * 3)(*[0., 0., gravity])
        #print 'gravity ', env.unwrapped.model.opt.gravity[:]
        #print 'wind ', env.unwrapped.model.opt.wind[:]
        #print 'magnetic ', env.unwrapped.model.opt.magnetic[:]
        #print 'density ', env.unwrapped.model.opt.density
        #print 'viscosity ', env.unwrapped.model.opt.viscosity
        #print 'body_com() ', env.unwrapped.get_body_com(env.unwrapped.model.body_names[1])
        #print 'body_comvel() ', env.unwrapped.get_body_comvel(env.unwrapped.model.body_names[1])
        #print 'body_cmat() ', env.unwrapped.get_body_xmat(env.unwrapped.model.body_names[1])

        for t in range(500):
            env.render()
            #print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            #time_steps += 1
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    #print 'time in seconds elapsed: ', time.time() - start_time, ' time steps: ', time_steps