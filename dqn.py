from atari_env import Environment
import tensorflow as tf
from observation_processing import preprocess
import numpy as np
import random

# Some helper functions for defining the network

def weight_variable(shape, name, initial_weight=None):
    if initial_weight:
        return tf.Variable(initial_weight, name=name)
    else:
        initial = tf.random_normal(shape, stddev=0.01)
        return tf.Variable(initial, name=name)


def bias_variable(shape, name, initial_weight=None):
    if initial_weight is None:
        initial = tf.constant(0.1, shape=shape, name=name)
        return tf.Variable(initial)
    else:
        return tf.Variable(initial_weight, name=name)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


class DQN:

    def __init__(self, load_path = 0):
        self.env = Environment("./Breakout.bin")
        self.replay_memory = []
        self.save_path = "/home/john/code/pythonfiles/my_dqn/saved_models/model.ckpt"
        self.load_path = load_path
        self.weights = self.initialize_network()




    def initialize_network(self):
        self.sess = tf.Session()
        self.input = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
        self.conv1_weight = weight_variable(shape=[8, 8, 4, 16], name='conv1_weight')
        self.conv1_bias = bias_variable(shape=[16], name='conv1_bias')
        self.conv1_layer = tf.nn.relu(conv2d(self.input, self.conv1_weight, 4) + self.conv1_bias)
        self.conv2_weight = weight_variable(shape=[4, 4, 16, 32], name='conv2_weight')
        self.conv2_bias = bias_variable(shape=[32], name='conv2_bias')
        self.conv2_layer = tf.nn.relu(conv2d(self.conv1_layer, self.conv2_weight, 2) + self.conv2_bias)
        self.conv2_layer_flattened = tf.reshape(self.conv2_layer, [-1, 9*9*32])
        self.fc1_weight = weight_variable(shape=[9*9*32, 256], name='fc1_weight')
        self.fc1_bias = bias_variable(shape=[256], name='fc1_bias')
        self.fc1_layer = tf.nn.relu(tf.matmul(self.conv2_layer_flattened, self.fc1_weight) + self.fc1_bias)
        self.fc2_weight = weight_variable(shape=[256, 4], name='fc2_weight')
        self.fc2_bias = bias_variable(shape=[4], name='fc2_bias')
        self.output = tf.matmul(self.fc1_layer, self.fc2_weight) + self.fc2_bias

        self.target = tf.placeholder(tf.float32, None)
        self.action_hot = tf.placeholder('float', [None,4])
        self.action_readout = tf.reduce_sum(tf.mul(self.output, self.action_hot), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(tf.sub(self.action_readout, self.target)))
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, decay=0.95, epsilon=0.01)
        self.gradients_and_vars = self.optimizer.compute_gradients(self.loss)
        self.gradients = [gravar[0] for gravar in self.gradients_and_vars]
        self.gradient_avgs = [tf.reduce_mean(grads) for grads in self.gradients]
        self.clipped_gradients = [(tf.clip_by_value(gv[0], -1., 1.), gv[1]) for gv in self.gradients_and_vars]
        self.train_operation = self.optimizer.apply_gradients(self.clipped_gradients)

        self.saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())

        self.load()

        weights = [self.conv1_weight, self.conv1_bias, self.conv2_weight, self.conv2_bias, self.fc1_weight,
                        self.fc1_bias, self.fc2_weight, self.fc2_bias]
        return weights

    def save(self):
        self.saver.save(self.sess, save_path=self.save_path)

    def load(self):
        if self.load_path != 0:
            self.saver.restore(self.sess, self.load_path)
            print("Model at path ", self.load_path, " loaded.")
        else:
            print("No model provided.")

    def update_replay_memory(self, tuple):
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > 1000000:
            self.replay_memory.pop(0)

    def test_network(self):
        # run for 20 episodes. Record step length, avg Q value, max reward, avg reward, loss, weight values for
        # each layer ...
        weights = self.sess.run(self.weights)
        layer1_weight_avg = np.average(np.absolute(weights[0]))
        layer1_bias_avg = np.average(np.absolute(weights[1]))
        layer2_weight_avg = np.average(np.absolute(weights[2]))
        layer2_bias_avg = np.average(np.absolute(weights[3]))
        layer3_weight_avg = np.average(np.absolute(weights[4]))
        layer3_bias_avg = np.average(np.absolute(weights[5]))
        layer4_weight_avg = np.average(np.absolute(weights[6]))
        layer4_bias_avg = np.average(np.absolute(weights[7]))
        weight_avgs = [layer1_weight_avg, layer1_bias_avg, layer2_weight_avg, layer2_bias_avg, layer3_weight_avg,
                         layer3_bias_avg, layer4_weight_avg, layer4_bias_avg]

        test_env = Environment('./Breakout.bin')
        total_reward = 0.
        total_steps = 0.
        Q_avg_total = 0.
        max_reward = 0.
        for ep in range(20):
            obs1 = test_env.reset()
            obs2 = test_env.step(test_env.sample_action())[0]
            obs3 = test_env.step(test_env.sample_action())[0]
            obs4, _, done = test_env.step(test_env.sample_action())
            obs1, obs2, obs3, obs4 = preprocess(obs1), preprocess(obs2), preprocess(obs3), preprocess(obs4)
            state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))
            episode_reward = 0.
            num_steps = 0.
            ep_Q_total = 0.
            done = False
            while not done:
                _, action, reward, new_state, obs1, obs2, obs3, obs4, Qval, done =\
                    self.true_step(0.05, state, obs2, obs3, obs4, test_env)
                state = new_state
                episode_reward += reward
                num_steps += 1.
                ep_Q_total += Qval
            max_reward = max(episode_reward, max_reward)
            ep_Q_avg = ep_Q_total/num_steps
            Q_avg_total += ep_Q_avg
            total_reward += episode_reward
            total_steps += num_steps

        avg_Q = Q_avg_total/20.
        avg_reward = total_reward/20.
        avg_steps = total_steps/20.
        print("Average Q-value: {}".format(avg_Q))
        print("Average episode reward: {}".format(avg_reward))
        print("Average number of steps: {}".format(avg_steps))
        print("Max reward over 20 episodes: {}".format(max_reward))

        return weight_avgs, avg_Q, avg_reward, max_reward, avg_steps




    # A helper that combines different parts of the step procedure
    def true_step(self, prob, state, obs2, obs3, obs4, env):

        Q_vals = self.sess.run(self.output, feed_dict={self.input: [state]})
        if random.uniform(0,1) > prob:
            step_action = Q_vals.argmax()
        else:
            step_action = env.sample_action()

        if prob > 0.1:
            prob -= 9.0e-7

        new_obs, step_reward, step_done = env.step(step_action)

        processed_obs = preprocess(new_obs)
        new_state = np.transpose([obs2, obs3, obs4, processed_obs], (1, 2, 0))

        # if done:
        #     reward -= 1

        return prob, step_action, step_reward, new_state, obs2, obs3, obs4, processed_obs, Q_vals.max(), step_done
