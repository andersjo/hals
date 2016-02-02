import random
from typing import List, Any

import numpy as np

from sentences import Sentence
from transition_parser.parallel_util import ParallelParse, SentenceBatch
from transition_parser.performance import measure_performance


class ThreadedTransitionParser:
    def __init__(self, transition_system, learner,
                 num_epochs=100, early_stop_after=10, p_explore=0.1,
                 train_batch_size=8, predict_batch_size=64):
        self.transition_system = transition_system
        self.learner = learner
        self.num_epochs = num_epochs
        self.p_explore = p_explore
        self.early_stop_after = early_stop_after

        self.num_correct = 0
        self.num_total = 0

        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size

        self.best_epoch = None
        self.best_epoch_score = 0

    def fit(self, train_sents, dev_sents=None):
        for epoch_i in range(1, self.num_epochs + 1):
            random.shuffle(train_sents)
            num_correct, num_total = self._fit_epoch(train_sents)

            print("Epoch {}: {}/{} = {}".format(epoch_i,
                                                num_correct, num_total,
                                                num_correct / num_total))

            if dev_sents:
                performance_dict = measure_performance(dev_sents, self.parse(dev_sents))
                perf_measure = performance_dict['uas']
                if perf_measure > self.best_epoch_score:
                    self.best_epoch = epoch_i
                    self.best_epoch_score = perf_measure
                else:
                    # We didn't get better: should we stop?
                    no_improvement_epochs = epoch_i - self.best_epoch
                    if no_improvement_epochs == self.early_stop_after:
                        print("No improvements for {} epochs. Stopping".format(self.early_stop_after))
                        return

                print("Dev: {}".format(performance_dict))

    def _fit_epoch(self, train_sents):
        trainer = ParallelTrain(train_sents, batch_size=self.train_batch_size, parser=self)
        parses = trainer.parse()

        performance_dict = measure_performance(train_sents, parses)
        return performance_dict['ua'], performance_dict['total']

    def parse(self, sentences):
        predictor = ParallelPredict(sentences, batch_size=self.predict_batch_size, parser=self)
        return predictor.parse()


class ParallelTrain(ParallelParse):
    def prepare_feed(self, sents: List[Sentence], states: List[Any]):
        return self.parser.learner.prepare_feed_batch(sents, states)

    def score_batch(self, batch: SentenceBatch):
        return self.parser.learner.score_actions_and_train_batch(batch.feed_dict, batch.allowed_list,
                                                                 batch.action_costs_list)

    def advance_batch(self, batch: SentenceBatch, batch_scores: List[np.ndarray]):
        t_sys = self.transition_system
        for i in range(len(batch.ids)):
            best_pred_action_index = np.argmax(batch_scores[i])
            allowed = batch.allowed_list[i]

            if random.random() < self.parser.p_explore:
                # Take the action predicted by the learner, even if it's an error
                action = allowed[best_pred_action_index]
            else:
                # Take the lowest cost action from the reference policy.
                # If multiple actions have the same cost, take the one
                # preferred by the learner.
                action_costs = batch.action_costs_list[i]
                min_cost_indices = np.where(action_costs == action_costs.min())[0]
                best_action_from_min_cost = batch_scores[i][min_cost_indices].argmax()
                action = np.array(allowed)[min_cost_indices][best_action_from_min_cost]

            t_sys.perform(batch.states[i], action)



class ParallelPredict(ParallelParse):
    def prepare_feed(self, sents: List[Sentence], states: List[Any]):
        return self.parser.learner.prepare_feed_batch(sents, states)

    def score_batch(self, batch: SentenceBatch):
        return self.parser.learner.score_actions_batch(batch.feed_dict, batch.allowed_list)

    def advance_batch(self, batch: SentenceBatch, batch_scores: List[np.ndarray]):
        for i in range(len(batch.ids)):
            best_pred_action_index = np.argmax(batch_scores[i])
            action = batch.allowed_list[i][best_pred_action_index]
            self.transition_system.perform(batch.states[i], action)
