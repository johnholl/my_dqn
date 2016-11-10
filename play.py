import tensorflow as tf
import numpy as np
from PIL import Image
import random
from atari_env import Environment
from observation_processing import preprocess
from trained_dqn import trained_DQN

# def weight_variable(shape, name, weight_value):
#     return tf.Variable(weight_value, name=name)
#
#
# def bias_variable(shape, name, bias_value):
#     initial = tf.constant(bias_value, shape=shape, name=name)
#     return tf.Variable(initial)
#
#
# def conv2d(x, W, stride):
#     return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')
#
#
# # Extract weights from npy file
# weights = np.load("./weights/weights_391.npy", encoding="latin1")
#
# # network
#
# sess = tf.Session()
# input = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
# conv1_weight = weight_variable(shape=[8, 8, 4, 16], name='conv1_weight', weight_value=weights[0])
# conv1_bias = bias_variable(shape=[16], name='conv1_bias', bias_value=weights[1])
# conv1_layer = tf.nn.relu(conv2d(input, conv1_weight, 4) + conv1_bias)
# conv2_weight = weight_variable(shape=[4, 4, 16, 32], name='conv2_weight', weight_value=weights[2])
# conv2_bias = bias_variable(shape=[32], name='conv2_bias', bias_value=weights[3])
# conv2_layer = tf.nn.relu(conv2d(conv1_layer, conv2_weight, 2) + conv2_bias)
# conv2_layer_flattened = tf.reshape(conv2_layer, [-1, 9*9*32])
# fc1_weight = weight_variable(shape=[9*9*32, 256], name='fc1_weight', weight_value=weights[4])
# fc1_bias = bias_variable(shape=[256], name='fc1_bias', bias_value=weights[5])
# fc1_layer = tf.nn.relu(tf.matmul(conv2_layer_flattened, fc1_weight) + fc1_bias)
# fc2_weight = weight_variable(shape=[256, 4], name='fc2_weight', weight_value=weights[6])
# fc2_bias = bias_variable(shape=[4], name='fc2_bias', bias_value=weights[7])
# output = tf.matmul(fc1_layer, fc2_weight) + fc2_bias
#
# env = Environment("./Breakout.bin")
# session = tf.Session()
# session.run(tf.initialize_all_variables())
# episode_step_count = []
#
# for episode in range(10):
#     obs1 = env.reset()
#     obs2 = env.step(env.sample_action())[0]
#     obs3 = env.step(env.sample_action())[0]
#     obs4, _, done = env.step(env.sample_action())
#     obs1, obs2, obs3, obs4 = preprocess(obs1), preprocess(obs2), preprocess(obs3), preprocess(obs4)
#     state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))
#     steps = 0
#     episode_reward = 0
#
#     while not env.ale.game_over():
#         env.render(rate=0.02)
#         Q_vals = session.run(output, feed_dict={input: [state]})
#         if random.random() > 0.05:
#             action = Q_vals.argmax()
#         else:
#             action = random.choice([0, 1, 2, 3])
#         obs1 = obs2
#         obs2 = obs3
#         obs3 = obs4
#         new_obs, reward, done = env.step(action)
#         episode_reward += reward
#         obs4 = preprocess(new_obs)
#         new_state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))
#         state = new_state
#         steps += 1
#
#     print("episode ", episode, " ran for ", steps, " steps")
#     print(episode_reward)

dqn = trained_DQN()
obs = dqn.env.reset()

for episode in range(10):
    obs1 = dqn.env.reset()
    obs2 = dqn.env.step(dqn.env.sample_action())[0]
    obs3 = dqn.env.step(dqn.env.sample_action())[0]
    obs4, _, done = dqn.env.step(dqn.env.sample_action())
    obs1, obs2, obs3, obs4 = preprocess(obs1), preprocess(obs2), preprocess(obs3), preprocess(obs4)
    state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))
    steps = 0
    episode_reward = 0

    while not dqn.env.ale.game_over():
        p, action, rew, new_state, obs1, obs2, obs3, obs4, maxQ, step_done = dqn.true_step(0.05, state, obs2, obs3, obs4, dqn.env)
        state = new_state
        dqn.env.render()