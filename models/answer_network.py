from dqn import DQN
import numpy as np


class Answer_network:

    def __init__(self):
        self.weights = []
        self.input = 0
        self.target = 0
        self.action_hot = 0
        self.train_op = 0
        self.sess = 0


    def update_weights(self):
        return self.weights


class DQN_answer_network(Answer_network):

    def __init__(self):
        Answer_network.__init__(self)
        self.dqn = DQN()
        self.weights = self.dqn.weights
        self.target_weights = self.weights
        self.input = self.dqn.input
        self.target = self.dqn.target
        self.action_hot = self.dqn.action_hot
        self.train_op = self.dqn.train_operation
        self.OUTPUT_SIZE = self.dqn
        self.sess = self.dqn.sess



    def update_weights(self, minibatch):
        batch_size = len(minibatch)
        next_states = np.array([m[3] for m in minibatch])
        feed_dict = {self.input: next_states}
        feed_dict.update(zip(self.weights, self.target_weights))
        q_vals = self.sess.run(self.output, feed_dict=feed_dict)
        max_q = q_vals.max(axis=1)
        target_q = np.zeros(batch_size)
        action_list = np.zeros((batch_size, self.dqn.OUTPUT_SIZE))
        for i in range(32):
            _, action_index, reward, _, terminal = minibatch[i]
            target_q[i] = reward
            if not terminal:
                target_q[i] = target_q[i] + 0.99*max_q[i]

            action_list[i][action_index] = 1.0

        states = [m[0] for m in minibatch]
        feed_dict = {dqn.input: np.array(states), dqn.target: target_q, dqn.action_hot: action_list}
        _, loss_val = dqn.sess.run(fetches=(dqn.train_operation, dqn.loss), feed_dict=feed_dict)


