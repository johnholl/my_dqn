from atari_env import Environment
import random
from observation_processing import preprocess, blacken_score
import numpy as np

env = Environment(rom="/home/john/code/pythonfiles/my_dqn/Breakout.bin")
obs = env.reset()
done = False
while not done:
    env.render()
    action = raw_input("enter an action: ")

    if action not in ["0", "1", "2", "3"]:
        print("Not a valid action. Try again. \n")
    else:
        obs, rew, done = env.step(action)
        obs = blacken_score(obs, obs.shape[0], obs.shape[1])
        print((obs == 0.0).sum())
