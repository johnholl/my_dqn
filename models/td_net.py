import tensorflow as tf
from observation_processing import preprocess
import numpy as np
import random
from atari_env import Environment
from layer_helpers import weight_variable, bias_variable, conv2d
from convolutional_network import Convnet

class DQN_TD_Net:

    def __init__(self, policy_p=0.05, rom='/home/john/code/pythonfiles/my_dqn/Breakout.bin'):
        self.rom = rom
        self.env = Environment(rom)
        self.sess = tf.Session()
        self.num_nodes = len(self.env.action_space) + 1
        self.nodes, self.node_targets, self.credit = self.initialize_question_network()
        self.num_nodes = len(self.nodes)
        self.answer_network = self.initialize_answer_network()
        self.input = self.answer_network.input
        self.weights = self.answer_network.weights
        self.output = self.answer_network.output
        self.loss, self.train_operation = self.initialize_learning_procedure()
        self.sess.run(tf.initialize_all_variables())
        self.replay_memory = []


    def initialize_question_network(self):
        dim = self.num_nodes
        nodes = tf.placeholder(tf.float32, shape=[None, dim])
        relation_matrix = np.zeros(shape=(dim, dim))

        # Default node relations to Q-learning
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    relation_matrix[i][j]=1.
                elif i == dim-1:
                    relation_matrix[i][j]=1.
        relations = tf.constant(relation_matrix)
        node_targets = tf.matmul(nodes, relations)

        # Credit matrix
        credit = tf.constant(relation_matrix)

        return nodes, node_targets, credit



    def initialize_answer_network(self):
        answer_network = Convnet(len(self.env.action_space), self.sess)
        return answer_network

    def initialize_learning_procedure(self):
        dim = self.num_nodes
        credit_vector = tf.placeholder(tf.float32, shape=[dim, 1])
        action_readout = tf.matmul(self.node_targets, credit_vector)
        loss = tf.reduce_mean(tf.square(tf.sub(action_readout, self.output)))
        optimizer = tf.train.RMSPropOptimizer(0.00025, decay=0.95, epsilon=0.01)
        gradients_and_vars = optimizer.compute_gradients(loss)
        gradients = [gravar[0] for gravar in gradients_and_vars]
        vars = [gravar[1] for gravar in gradients_and_vars]
        clipped_gradients = tf.clip_by_global_norm(gradients, 1.)[0]
        train_operation = optimizer.apply_gradients(zip(clipped_gradients, vars))
        return loss, train_operation


    def update_replay_memory(self, tuple):
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > 1000000:
            self.replay_memory.pop(0)


    def update_answer_weights(self):
        return 0

    def test_network(self):
        return 0

    def true_step(self, prob, state, obs2, obs3, obs4, env):

        output = self.sess.run(self.output, feed_dict={self.input: [np.array(state)]})
        Q_vals = output[0][:self.ACTION_SIZE]
        if random.uniform(0, 1) > prob:
            step_action = Q_vals.argmax()
        else:
            step_action = env.sample_action()

        if prob > 0.1:
            prob -= 9.0e-7

        new_obs, step_reward, step_done = env.step(step_action)

        processed_obs = preprocess(new_obs)
        new_state = np.transpose([obs2, obs3, obs4, processed_obs], (1, 2, 0))

        return prob, step_action, step_reward, new_state, obs2, obs3, obs4, processed_obs, Q_vals.max(), step_done


class TD_net:

    def __init__(self, relations, credits, rom='/home/john/code/pythonfiles/my_dqn/Breakout.bin'):
        self.rom = rom
        self.sess = tf.Session()
        self.env = Environment(self.rom)
        self.relations = tf.constant(relations, tf.float32)
        self.credits = credits
        self.dim = len(relations)
        self.nodes, self.node_targets = self.initialize_question_network()
        self.answer_network = self.initialize_answer_network()
        self.loss, self.train_operation, self.credit_vector = self.initialize_learning_procedure()
        self.sess.run(tf.initialize_all_variables())
        self.replay_memory = []

    def initialize_question_network(self):
        nodes = tf.placeholder(tf.float32, shape=[None, self.dim])
        node_targets = tf.matmul(nodes, self.relations)
        return nodes, node_targets

    def initialize_answer_network(self):
        answer_network = Convnet(self.dim, self.sess)
        return answer_network

    def initialize_learning_procedure(self):
        credit_vector = tf.placeholder(tf.float32, shape=[self.dim, 1])
        action_readout = tf.matmul(self.node_targets, credit_vector)
        loss = tf.reduce_mean(tf.square(tf.sub(action_readout, self.answer_network.output)))
        optimizer = tf.train.RMSPropOptimizer(0.00025, decay=0.95, epsilon=0.01)
        gradients_and_vars = optimizer.compute_gradients(loss)
        gradients = [gravar[0] for gravar in gradients_and_vars]
        vars = [gravar[1] for gravar in gradients_and_vars]
        clipped_gradients = tf.clip_by_global_norm(gradients, 1.)[0]
        train_operation = optimizer.apply_gradients(zip(clipped_gradients, vars))
        return loss, train_operation, credit_vector

    def true_step(self, prob, state, obs2, obs3, obs4, env):
        output = self.sess.run(self.answer_network.output, feed_dict={self.answer_network.input: [np.array(state)]})
        Q_vals = output[0][-len(self.env.action_space):]
        if random.uniform(0, 1) > prob:
            step_action = Q_vals.argmax()
        else:
            step_action = env.sample_action()

        if prob > 0.1:
            prob -= 9.0e-7

        new_obs, step_reward, step_done = env.step(step_action)

        processed_obs = preprocess(new_obs)
        new_state = np.transpose([obs2, obs3, obs4, processed_obs], (1, 2, 0))

        return prob, step_action, step_reward, new_state, obs2, obs3, obs4, processed_obs, Q_vals.max(), step_done

    def test_network(self):
        # run for 20 episodes. Record step length, avg Q value, max reward, avg reward, loss, weight values for
        # each layer ...
        weights = self.sess.run(self.answer_network.weights)
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

    def update_replay_memory(self, tuple):
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > 1000000:
            self.replay_memory.pop(0)




