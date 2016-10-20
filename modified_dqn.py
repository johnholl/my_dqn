from atari_env import Environment
import tensorflow as tf
from observation_processing import preprocess
import numpy as np
import random
from layer_helpers import weight_variable, bias_variable, conv2d

# below is a TD network that takes normal DQN and adds in random weighted pixel sum predictions


class TD_DQN:

    def __init__(self, num_predictions, rom='Breakout.bin'):
        self.rom = rom
        self.env = Environment(rom=self.rom)
        self.OUTPUT_SIZE = len(self.env.action_space)*(1 + num_predictions)
        self.weights = self.initialize_network()
        self.replay_memory = []



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
        self.fc2_weight = weight_variable(shape=[256, self.OUTPUT_SIZE], name='fc2_weight')
        self.fc2_bias = bias_variable(shape=[self.OUTPUT_SIZE], name='fc2_bias')
        self.output = tf.matmul(self.fc1_layer, self.fc2_weight) + self.fc2_bias

        self.target = tf.placeholder(tf.float32, None)
        self.action_hot = tf.placeholder('float', [None, self.OUTPUT_SIZE])
        self.action_readout = tf.reduce_sum(tf.mul(self.output, self.action_hot), reduction_indices=1)
        self.loss = tf.reduce_mean(.5*tf.square(tf.sub(self.action_readout, self.target)))
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, decay=0.95, epsilon=0.01)
        self.gradients_and_vars = self.optimizer.compute_gradients(self.loss)
        self.gradients = [gravar[0] for gravar in self.gradients_and_vars]
        self.vars = [gravar[1] for gravar in self.gradients_and_vars]
        self.clipped_gradients = tf.clip_by_global_norm(self.gradients, 1.)[0]
        self.train_operation = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.vars))
        self.sess.run(tf.initialize_all_variables())

        weights = [self.conv1_weight, self.conv1_bias, self.conv2_weight, self.conv2_bias, self.fc1_weight,
                   self.fc1_bias, self.fc2_weight, self.fc2_bias]
        return weights

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

        test_env = Environment(self.rom)
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
            # done = False
            while not test_env.ale.game_over():
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

        Q_vals = self.sess.run(self.output, feed_dict={self.input: [np.array(state)]})
        if random.uniform(0,1) > prob:
            step_action = Q_vals.argmax()
        else:
            step_action = env.sample_action()

        if prob > 0.1:
            prob -= 9.0e-7

        new_obs, step_reward, step_done = env.step(step_action)

        processed_obs = preprocess(new_obs)
        new_state = np.transpose([obs2, obs3, obs4, processed_obs], (1, 2, 0))

        return prob, step_action, step_reward, new_state, obs2, obs3, obs4, processed_obs, Q_vals.max(), step_done