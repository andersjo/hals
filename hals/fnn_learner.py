from typing import List

import numpy as np
import tensorflow as tf

from arc_eager import ArcEagerParseState
from feature import LearnerModel
from sentences import Sentence


class FnnLearner:
    def __init__(self, model: LearnerModel, optimizer=tf.train.GradientDescentOptimizer(0.01), clip_gradient=True):
        self.model = model

        # Add a train step
        # self.train_step = optimizer.minimize(model.loss)

        grads_and_vars = optimizer.compute_gradients(model.loss)
        if clip_gradient:
            clipped_grads_and_vars = []
            for grad, var in grads_and_vars:
                clipped_grad = tf.clip_by_value(grad, -1, 1)
                clipped_grads_and_vars.append((clipped_grad, var))

            grads_and_vars = clipped_grads_and_vars

        self.train_step = optimizer.apply_gradients(grads_and_vars)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def score_actions(self, sent, state, allowed):
        feed_dict = self.model.prepare_feed(sent, state)

        all_scores = self.session.run(self.model.logits, feed_dict)
        all_scores = all_scores.reshape(-1)

        return np.array([all_scores[i] for i in allowed])

    def score_actions_batch(self, feed_dict, allowed_list) -> List[np.ndarray]:
        # For now we return a list of arrays with the allowed scores.
        # Another option is to return a dense array with scores for all possibilities.
        # Or perhaps sparse array with scores for only those allowed.
        feed_dict[self.model.dropout_keep_p_ph] = 1.0
        all_scores = self.session.run(self.model.logits, feed_dict)

        return _allowed_scores_as_list(all_scores, allowed_list)

    def prepare_feed_batch(self, sents, states):
        return self.model.prepare_feed_batch(sents, states)

    def score_actions_and_train(self, sent: Sentence, state: ArcEagerParseState, allowed, costs):
        # Create gold y
        y = np.zeros(self.model.num_classes, dtype=np.float32)
        min_cost = costs.min()
        for i, cost in zip(allowed, costs):
            if cost == min_cost:
                y[i] = 1

        # Build inputs
        feed_dict = self.model.prepare_feed(sent, state)
        feed_dict[self.model.gold_ph] = y.reshape(1, -1)
        feed_dict[self.model.dropout_keep_p_ph] = 0.5

        all_scores, _ = self.session.run([self.model.logits, self.train_step], feed_dict)

        all_scores = all_scores.reshape(-1)

        return all_scores[allowed]

    def score_actions_and_train_batch(self, feed_dict, allowed_list, costs_list):
        # Create gold y
        y = np.zeros([len(costs_list), self.model.num_classes], dtype=np.float32)

        for i in range(len(costs_list)):
            costs = costs_list[i]
            allowed = allowed_list[i]

            min_cost = costs.min()
            for j, cost in zip(allowed, costs):
                if cost == min_cost:
                    y[i, j] = 1.0

        feed_dict[self.model.gold_ph] = y
        feed_dict[self.model.dropout_keep_p_ph] = 0.5

        all_scores, _ = self.session.run([self.model.logits, self.train_step], feed_dict)

        return _allowed_scores_as_list(all_scores, allowed_list)


def _allowed_scores_as_list(all_scores, allowed_list):
    allowed_scores = []
    for i in range(all_scores.shape[0]):
        allowed_scores.append(np.array([all_scores[i, j]
                                        for j in allowed_list[i]]))

    return allowed_scores
