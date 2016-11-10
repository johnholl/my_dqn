import tensorflow as tf
import numpy as np
from layer_helpers import conv2d, weight_variable, bias_variable
from trained_dqn import trained_DQN
from observation_processing import preprocess
import random

# network


class prediction_network:

    def __init__(self, num_pred):
        self.sess = tf.Session()
        self.input = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
        self.conv1_weight = weight_variable(shape=[8, 8, 4, 16], name='conv1_weight')
        self.conv1_bias = bias_variable(shape=[16], name='conv1_bias')
        conv1_layer = tf.nn.relu(conv2d(self.input, self.conv1_weight, 4) + self.conv1_bias)
        self.conv2_weight = weight_variable(shape=[4, 4, 16, 32], name='conv2_weight')
        self.conv2_bias = bias_variable(shape=[32], name='conv2_bias')
        conv2_layer = tf.nn.relu(conv2d(conv1_layer, self.conv2_weight, 2) + self.conv2_bias)
        conv2_layer_flattened = tf.reshape(conv2_layer, [-1, 9*9*32])
        self.fc1_weight = weight_variable(shape=[9*9*32, 256], name='fc1_weight')
        self.fc1_bias = bias_variable(shape=[256], name='fc1_bias')
        fc1_layer = tf.nn.relu(tf.matmul(conv2_layer_flattened, self.fc1_weight) + self.fc1_bias)
        self.fc2_weight = weight_variable(shape=[256, num_pred], name='fc2_weight')
        self.fc2_bias = bias_variable(shape=[20], name='fc2_bias')
        self.output = tf.matmul(fc1_layer, self.fc2_weight) + self.fc2_bias

        self.target = tf.placeholder(tf.float32, shape=[None, num_pred])
        self.loss = tf.reduce_mean(tf.reduce_sum(.5*tf.square(tf.sub(self.output, self.target)), reduction_indices=1))

        self.optimizer = tf.train.RMSPropOptimizer(0.00025, decay=0.95, epsilon=0.01)
        self.gradients_and_vars = self.optimizer.compute_gradients(self.loss)
        self.gradients = [gravar[0] for gravar in self.gradients_and_vars]
        self.vars = [gravar[1] for gravar in self.gradients_and_vars]
        self.clipped_gradients = tf.clip_by_global_norm(self.gradients, 1.)[0]
        self.train_operation = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.vars))
        self.sess.run(tf.initialize_all_variables())

        self.weights = [self.conv1_weight, self.conv1_bias, self.conv2_weight, self.conv2_bias, self.fc1_weight,
                        self.fc1_bias, self.fc2_weight, self.fc2_bias]



BATCH_SIZE = 32
projections = np.random.rand(4*84*84, 20)/(255.*84.*84.*4.*20.)


print(np.shape(projections))

dqn = trained_DQN()
replay_memory = []
prediction_network = prediction_network(20)
prediction_weights = prediction_network.sess.run(prediction_network.weights)
total_step = 0
loss_array = []

while total_step < 1000000:
    obs1 = dqn.env.reset()
    obs2 = dqn.env.step(dqn.env.sample_action())[0]
    obs3 = dqn.env.step(dqn.env.sample_action())[0]
    obs4, _, done = dqn.env.step(dqn.env.sample_action())
    obs1, obs2, obs3, obs4 = preprocess(obs1), preprocess(obs2), preprocess(obs3), preprocess(obs4)
    state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))
    steps = 0
    episode_reward = 0
    episode_count = 0
    while not dqn.env.ale.game_over():
        p, action, rew, new_state, obs1, obs2, obs3, obs4, maxQ, step_done = dqn.true_step(0.05, state, obs2, obs3, obs4, dqn.env)
        replay_memory.append((state, action, new_state, step_done))
        state = new_state

        if len(replay_memory) > 100000 and total_step % 4:
            minibatch = random.sample(replay_memory, BATCH_SIZE)

            next_states = np.array([m[2] for m in minibatch])
            terminal = np.array([m[3] for m in minibatch])
            states = np.array([m[0] for m in minibatch])
            shaped_next_states = np.reshape(next_states, [-1, 4*84*84])
            next_proj_values = np.dot(shaped_next_states, projections)
            print(np.shape(next_proj_values))
            next_predictions = prediction_network.sess.run(prediction_network.output, feed_dict={prediction_network.input: next_states})
            prediction_target = np.zeros(shape=[BATCH_SIZE, 20])
            for i in range(BATCH_SIZE):
                prediction_target[i] = next_proj_values[i]
                if not terminal[i]:
                    prediction_target[i] += next_predictions[i]

            feed_dict = {prediction_network.input: states, prediction_network.target: prediction_target}
            _, loss = prediction_network.sess.run((prediction_network.train_operation, prediction_network.loss), feed_dict=feed_dict)
            loss_array.append(loss)

        if total_step % 10000 == 0:
            np.save('loss_values', np.sum(loss_array[-100:])/100.)
            weights = prediction_network.sess.run(prediction_network.weights)
            np.save('weights_' + str(int(total_step/50000)), weights)

        total_step += 1

    episode_count +=1
    print("episode count: ", episode_count, ". last 100 step average loss: ", np.sum(loss_array[-100:])/100.)

