from atari_env import Environment
import random
from observation_processing import preprocess
import numpy as np

env = Environment(rom="/home/john/code/pythonfiles/my_dqn/beam_rider.bin")
print(env.action_space)
obs = env.reset()
done = False
while not done:
    #action = random.choice([0,1,2,3])
    action = raw_input("enter an action")
    obs, rew, done = env.step(action)
    obs = preprocess(obs)
    print(np.max(obs))
    print(np.min(obs))
