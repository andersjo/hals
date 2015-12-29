import random

import numpy as np


class RandomLearner:
    def __init__(self, num_labels):
        self.num_labels = num_labels

    def score(self, sent, state, allowed):
        return np.array([random.random() for _ in allowed])

    def score_and_update(self, sent, state, allowed, costs):
        return self.score(sent, state, allowed)
