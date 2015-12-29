import random

import numpy as np
import tensorflow as tf

from arc_eager import ArcEagerParseState
from feature import LearnerModel
from sentences import Sentence


class FnnLearner:
    def __init__(self, model: LearnerModel, optimizer=tf.train.GradientDescentOptimizer(0.01)):
        self.model = model

        # Add a train step
        self.train_step = optimizer.minimize(model.loss)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def score(self, sent, state, allowed):
        feed_dict = self.model.prepare_feed(sent, state)

        all_scores = self.session.run(self.model.logits, feed_dict)
        all_scores = all_scores.reshape(-1)

        return np.array([all_scores[i] for i in allowed])

    def score_and_update(self, sent: Sentence, state: ArcEagerParseState, allowed, costs):
        # Create gold y
        y = np.zeros(self.model.num_classes, dtype=np.float32)
        min_cost = costs.min()
        for i, cost in zip(allowed, costs):
            if cost == min_cost:
                y[i] = 1

        # Build inputs
        feed_dict = self.model.prepare_feed(sent, state)
        feed_dict[self.model.gold_ph] = y.reshape(1, -1)

        all_scores, _ = self.session.run([self.model.logits, self.train_step], feed_dict)

        all_scores = all_scores.reshape(-1)

        return all_scores[allowed]
