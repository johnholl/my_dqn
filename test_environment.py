from atari_env import Environment
import random

env = Environment(rom="/home/john/code/pythonfiles/my_dqn/Breakout.bin")
obs = env.reset()
done = False
while not done:
    #action = random.choice([0,1,2,3])
    action = raw_input("enter an action")
    obs, rew, done = env.step(action)
    print(rew)
    # raw_input("Press Enter to continue...")
    env.render(0.01)
