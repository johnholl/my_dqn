import tensorflow as tf
from observation_processing import preprocess
import numpy as np
import random
from atari_env import Environment
from layer_helpers import weight_variable, bias_variable, conv2d
from convolutional_network import Convnet



class TD_Net:

    def __init__(self, policy_p=0.05, rom='/home/john/code/pythonfiles/my_dqn/Breakout.bin'):
        self.rom = rom
        self.env = Environment(rom)
        self.sess = tf.Session()
        self.num_nodes = len(self.env.action_space) + 1
        self.nodes, self.node_targets, self.credit = self.initialize_question_network()
        self.num_nodes = len(self.nodes)
        self.sess.run(tf.initialize_all_variables())
        self.answer_network = self.initialize_answer_network()
        self.input = self.answer_network.input
        self.weights = self.answer_network.weights
        self.output = self.answer_network.output
        self.loss = self.initialize_learning_procedure()

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

        return loss

    def update_answer_weights(self):

        return 0

    def test_network(self):
        return 0


