import gym
import gym_extensions

env = gym.make('CartPoleContextual-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
    print env.unwrapped.context_space_info()
    context_vect = [0.1, 0.1, 0.1, 0.1]
    print 'context before: ', env.unwrapped.context 
    env.unwrapped.change_context(context_vect)
    print 'context after:  ', env.unwrapped.context
    


env = gym.make('PendulumContextual-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
    print env.unwrapped.context_space_info()
    context_vect = [0.1, 0.1]
    print 'context before: ', env.unwrapped.context 
    env.unwrapped.change_context(context_vect)
    print 'context after:  ', env.unwrapped.context