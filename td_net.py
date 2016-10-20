from dqn import DQN
import tensorflow as tf
from observation_processing import preprocess
import numpy as np
import random



class TD_Net:

    def __init__(self):
        self.dqn = DQN()
        self.weights = self.initialize_question_network()

        return

