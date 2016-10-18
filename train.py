from dqn import DQN
from observation_processing import preprocess
import numpy as np
import random
from threading import Thread


def save_data(qnet, lossarr, prob, learn_data, weightarr, targ_weight):
    weight_avgs, avg_Q, avg_rewards, max_reward, avg_steps = qnet.test_network()
    learn_data.append([total_steps, avg_Q, avg_rewards, max_reward, avg_steps,
                          np.mean(lossarr[-100]), prob])
    weightarr.append(weight_avgs)
    np.save('learning_data', learn_data)
    np.save('weight_averages', weightarr)
    np.save('weights_' + str(int(total_steps/50000)), targ_weight)




dqn = DQN()

target_weights = dqn.sess.run(dqn.weights)
episode_step_count = []
total_steps = 1.
prob = 1.0
learning_data = []
weight_average_array = []
loss_vals = []
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
            minibatch = random.sample(dqn.replay_memory, 32)
            next_states = np.array([m[3] for m in minibatch])/255.
            feed_dict = {dqn.input: next_states}
            feed_dict.update(zip(dqn.weights, target_weights))
            q_vals = dqn.sess.run(dqn.output, feed_dict=feed_dict)
            max_q = q_vals.max(axis=1)
            target_q = np.zeros(32)
            action_list = np.zeros((32,4))
            for i in range(32):
                _, action_index, reward, _, terminal = minibatch[i]
                target_q[i] = reward
                if not terminal:
                    target_q[i] = target_q[i] + 0.99*max_q[i]

                action_list[i][action_index] = 1.0

            states = [m[0] for m in minibatch]
            feed_dict = {dqn.input: np.array(states)/255., dqn.target: target_q, dqn.action_hot: action_list}
            _, loss_val = dqn.sess.run(fetches=(dqn.train_operation, dqn.loss), feed_dict=feed_dict)
            loss_vals.append(loss_val)

        if total_steps % 10000 == 0:
            target_weights = dqn.sess.run(dqn.weights)

        if total_steps % 50000 == 0:
            testing_thread = Thread(target=save_data, args=(dqn, loss_val, prob,
                                                            learning_data, weight_average_array, target_weights))
            testing_thread.start()

            # weight_avgs, avg_Q, avg_rewards, max_reward, avg_steps = dqn.test_network()
            # learning_data.append([total_steps, avg_Q, avg_rewards, max_reward, avg_steps,
            #                       np.mean(loss_vals[-100]), prob])
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
