from dqn import DQN
from observation_processing import preprocess
import numpy as np
import random
import tensorflow as tf


dqn = DQN()

summary_writer = tf.summary.FileWriter("/home/john/tmp/dqn")

target_weights = dqn.sess.run(dqn.weights)
replay_memory = []
episode_step_count = []
total_steps = 1.
prob = 1.0
learning_data = []
weight_average_array = []
loss_vals = []
episode_number = 0

while total_steps < 20000000:
    obs1 = np.squeeze(dqn.env.reset())
    obs2 = np.squeeze(dqn.env.step(dqn.sample_action())[0])
    obs3 = np.squeeze(dqn.env.step(dqn.sample_action())[0])
    obs4, _, done, info = dqn.env.step(dqn.sample_action())
    obs4 = np.squeeze(obs4)
    obs1, obs2, obs3, obs4 = preprocess(obs1), preprocess(obs2), preprocess(obs3), preprocess(obs4)
    state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))
    steps = 0.
    episode_reward = 0.

    while not done:
        prob, action, reward, new_state, obs1, obs2, obs3, obs4, _, done = dqn.true_step(prob, state, obs2, obs3, obs4, dqn.env)
        dqn.update_replay_memory((state, action, reward, new_state, done))
        state = new_state
        episode_reward += reward


        if len(dqn.replay_memory) >= 100 and total_steps % 4 == 0:
            # here is where the training procedure takes place
            # compute the one step q-values w.r.t. old weights (ie y in the loss function (y-Q(s,a,0))^2)
            # Also defines the one-hot readout action vectors
            minibatch = random.sample(dqn.replay_memory, 32)
            next_states = [m[3] for m in minibatch]
            feed_dict = {dqn.input: next_states}
            feed_dict.update(zip(dqn.weights, target_weights))
            q_vals = dqn.sess.run(dqn.output, feed_dict=feed_dict)
            max_q = q_vals.max(axis=1)
            target_q = np.zeros(32)
            action_list = np.zeros((32,6))
            for i in range(32):
                _, action_index, reward, _, terminal = minibatch[i]
                # print(action_index)
                # print(reward)
                # print(terminal)
                target_q[i] = reward
                if not terminal:
                    target_q[i] = target_q[i] + 0.99*max_q[i]

                action_list[i][action_index] = 1.0

            if total_steps % 11:
                fetches = (dqn.summary_op, dqn.train_operation, dqn.loss, dqn.global_step)
            else:
                fetches = (dqn.train_operation, dqn.loss, dqn.global_step)

            states = [m[0] for m in minibatch]
            feed_dict = {dqn.input: states, dqn.target: target_q, dqn.action_hot: action_list}
            fetched = dqn.sess.run(fetches=fetches, feed_dict=feed_dict)
            loss_vals.append(fetched[1])

            if total_steps % 11:
                summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
                summary_writer.flush()

        if total_steps % 10000 == 0:
            target_weights = dqn.sess.run(dqn.weights)

        if total_steps % 50000 == 0:
            weight_avgs, avg_Q, avg_rewards, max_reward, avg_steps = dqn.test_network()
            learning_data.append([total_steps, avg_Q, avg_rewards, max_reward, avg_steps,
                                  np.mean(loss_vals[-100]), prob])
            weight_average_array.append(weight_avgs)
            np.save('learning_data', learning_data)
            np.save('weight_averages', weight_average_array)
            dqn.save()
            np.save('weights_' + str(int(total_steps/50000)), target_weights)

        total_steps += 1
        steps += 1

        if done:
            summary = tf.Summary()
            summary.value.add(tag='episode_reward', simple_value=float(episode_reward))
            summary.value.add(tag='reward_per_timestep', simple_value=float(episode_reward)/float(steps))
            summary_writer.add_summary(summary, dqn.sess.run(dqn.global_step))
            summary_writer.flush()
            episode_number += 1
            break

    episode_step_count.append(steps)
    mean_steps = np.mean(episode_step_count[-100:])
    print("Training episode = {}, Total steps = {}, Last 100 mean steps = {}, Episode reward = {}"
          .format(episode_number, total_steps, mean_steps, episode_reward))
