from models.td_net import TD_net
from observation_processing import preprocess
import numpy as np
import random
from threading import Thread
import tensorflow as tf
import sys

BATCH_SIZE = 32
NUM_PROJECTIONS = 10
ACTION_SPACE_SIZE = 4
NUM_NODES = NUM_PROJECTIONS*ACTION_SPACE_SIZE + 1
# create 10 random vectors to project the observation
projections = np.random.rand(NUM_PROJECTIONS, 4*84*84)/(255.*84.*84.*4.)


def save_data(qnet, lossarr, lossarr_pred, lossarr_q, prob, learn_data, weightarr, targ_weight):
    weight_avgs, avg_Q, avg_rewards, max_reward, avg_steps = qnet.test_network()
    learn_data.append([total_steps, avg_Q, avg_rewards, max_reward, avg_steps,
                          np.mean(lossarr[-100:]), np.mean(lossarr_pred[-100:]), np.mean(lossarr_q[-100:]), prob])
    weightarr.append(weight_avgs)
    np.save('learning_data', learn_data)
    np.save('weight_averages', weightarr)
    np.save('weights_' + str(int(total_steps/50000)), targ_weight)

## Form the relation and credit matrix for the TD network
relation_matrix = np.zeros(shape=(NUM_NODES, NUM_NODES))

# Default node relations to Q-learning
for i in range(NUM_NODES):
    for j in range(NUM_NODES):
        if i == j:
            relation_matrix[i][j] = 1.
        elif i == NUM_NODES-1:
            relation_matrix[i][j] = 1.

credit_matrix = relation_matrix

dqn = TD_net(relations=relation_matrix, credits=credit_matrix, rom='Breakout.bin')

target_weights = dqn.sess.run(dqn.answer_network.weights)
episode_step_count = []
total_steps = 1.
prob = 1.0
learning_data = []
weight_average_array = []
loss_vals = []
loss_vals_pred = []
loss_vals_q = []
episode_number = 0

while total_steps < 20000000:
    obs1 = dqn.env.reset()
    obs2 = dqn.env.step(dqn.env.sample_action())[0]
    obs3 = dqn.env.step(dqn.env.sample_action())[0]
    obs4, _, terminal = dqn.env.step(dqn.env.sample_action())
    obs1, obs2, obs3, obs4 = preprocess(obs1), preprocess(obs2), preprocess(obs3), preprocess(obs4)
    state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))
    steps = 0

    while not dqn.env.ale.game_over():
        prob, action, reward, new_state, obs1, obs2, obs3, obs4, _, terminal = dqn.true_step(prob, state, obs2, obs3, obs4, dqn.env)
        dqn.update_replay_memory((state, action, reward, new_state, terminal))
        state = new_state

        if len(dqn.replay_memory) >= 40000 and total_steps % 4 == 0:
            # here is where the training procedure takes place
            # compute the one step q-values w.r.t. old weights (ie y in the loss function (y-Q(s,a,0))^2)
            # Also defines the one-hot readout action vectors
            minibatch = random.sample(dqn.replay_memory, BATCH_SIZE)
            next_states = np.array([m[3] for m in minibatch])
            feed_dict = {dqn.answer_network.input: next_states}
            feed_dict.update(zip(dqn.answer_network.weights, target_weights))
            outputs = dqn.sess.run(dqn.answer_network.output, feed_dict=feed_dict)
            targets = dqn.sess.run(dqn.node_targets, feed_dict={dqn.nodes : outputs})
            states = [m[0] for m in minibatch]
            feed_dict = {dqn.answer_network.input: states, dqn.node_targets: targets, dqn.credit_vector: credits[action]}
            dqn.sess.run(dqn.train_operation, feed_dict=feed_dict)

        if total_steps % 10000 == 0:
            target_weights = dqn.sess.run(dqn.answer_network.weights)

        if total_steps % 50000 == 0:
            testing_thread = Thread(target=save_data, args=(dqn, loss_vals, loss_vals_pred, loss_vals_q, prob,
                                                            learning_data, weight_average_array, target_weights))
            testing_thread.start()

            # weight_avgs, avg_Q, avg_rewards, max_reward, avg_steps = dqn.test_network()
            # learning_data.append([total_steps, avg_Q, avg_rewards, max_reward, avg_steps,
            #                       np.mean(loss_vals[-100:]), prob])
            # weight_average_array.append(weight_avgs)
            # np.save('learning_data', learning_data)
            # np.save('weight_averages', weight_average_array)
            # np.save('weights_' + str(int(total_steps/50000)), target_weights)

        total_steps += 1
        steps += 1


    episode_number += 1


    episode_step_count.append(steps)
    mean_steps = np.mean(episode_step_count[-100:])
    print("Training episode = {}, Total steps = {}, Last 100 mean steps = {}"
          .format(episode_number, total_steps, mean_steps))
