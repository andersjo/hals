import random
import numpy as np

from sentences import Sentence


class SerialParser:
    def __init__(self, transition_system, learner, num_epochs=100):
        self.transition_system = transition_system
        self.learner = learner
        self.num_epochs = num_epochs
        self.p_explore = 0.1

        self.num_correct = 0
        self.num_total = 0

    def fit(self, train_sents, dev_sents=None):
        for epoch_i in range(1, self.num_epochs + 1):
            random.shuffle(train_sents)
            self.num_correct = 0
            self.num_total = 0

            for sent in train_sents:
                self._train_sent(sent)

            print("Epoch {}: {}/{} = {}".format(epoch_i,
                                                self.num_correct, self.num_total,
                                                self.num_correct / self.num_total))

            if dev_sents:
                print("Dev: {}".format(self.score_parses(dev_sents)))

    def parse(self, sentences):
        for sent in sentences:
            yield self._parse_sent(sent)

    # TODO refactor to extract the scoring part. Maybe introduce some class representing a finished parse.
    def score_parses(self, sentences):
        correct_labels = 0
        correct_edges = 0
        total = 0

        for sent, parsed_sent in zip(sentences, self.parse(sentences)):
            heads, labels = parsed_sent
            total += len(heads)
            num_unlabeled, num_labeled = sent.num_correct(heads, labels)
            correct_edges += num_unlabeled
            correct_labels += num_labeled

        return {
            'ua': correct_edges,
            'la': correct_labels,
            'total': total,
            'uas': correct_edges / total,
            'las': correct_labels / total
        }

    def _parse_sent(self, sent):
        t_sys = self.transition_system
        state = t_sys.state(len(sent))

        while not t_sys.is_final(state):
            allowed = t_sys.allowed(state)

            score_pred_actions = self.learner.score_actions(sent, state, allowed)
            best_pred_action_index = np.argmax(score_pred_actions)
            action = allowed[best_pred_action_index]
            t_sys.perform(state, action)

        return t_sys.extract_parse(state)

    def _train_sent(self, sent: Sentence):
        t_sys = self.transition_system
        ref_policy = t_sys.reference_policy()

        state = t_sys.state(len(sent))

        while not t_sys.is_final(state):
            allowed = t_sys.allowed(state)
            assert len(allowed)
            action_costs = ref_policy(state, sent, allowed)
            min_cost = action_costs.min()

            score_pred_actions = self.learner.score_actions_and_train(sent, state, allowed, action_costs)
            best_pred_action_index = np.argmax(score_pred_actions)

            self.num_total += 1
            if action_costs[best_pred_action_index] == min_cost:
                self.num_correct += 1

            # Pick action
            if random.random() < self.p_explore:
                # Take the action predicted by the learner, even if it's an error
                action = allowed[best_pred_action_index]
            else:
                # Take the lowest cost action from the reference policy.
                # If multiple actions have the same cost, take the one
                # preferred by the learner.
                min_cost_indices = np.where(action_costs == min_cost)
                action = allowed[score_pred_actions[min_cost_indices].argmax()]

            t_sys.perform(state, action)