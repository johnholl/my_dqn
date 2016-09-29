from atari_env import Environment
import tensorflow as tf

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

    def __init__(self):
        self.env = Environment("./Breakout.bin")
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
        self.fc2_weight = weight_variable(shape=[256, 6], name='fc2_weight')
        self.fc2_bias = bias_variable(shape=[6], name='fc2_bias')
        self.output = tf.matmul(self.fc1_layer, self.fc2_weight) + self.fc2_bias

        self.target = tf.placeholder(tf.float32, None)
        self.action_hot = tf.placeholder('float', [None,6])
        self.action_readout = tf.reduce_sum(tf.mul(self.output, self.action_hot), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(tf.sub(self.action_readout, self.target)))
        self.train_operation = tf.train.RMSPropOptimizer(0.00025, decay=0.95, epsilon=1e-6).minimize(self.loss)

        weights = [self.conv1_weight, self.conv1_bias, self.conv2_weight, self.conv2_bias, self.fc1_weight,
                        self.fc1_bias, self.fc2_weight, self.fc2_bias]
        return weights

    def update_replay_memory(self, tuple):
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > 1000000:
            self.replay_memory.pop(0)
