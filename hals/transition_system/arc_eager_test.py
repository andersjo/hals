from copy import copy, deepcopy
import numpy as np
from unittest import TestCase

from transition_system.arc_eager import ArcEager, ArcEagerDynamicOracle


def generate_all_projective_parses(size):
    arc_eager = ArcEager(1)
    initial = arc_eager.state(size)
    stack = []
    stack.append(initial)

    parses = set()
    while len(stack):
        state = stack.pop()
        if arc_eager.is_final(state):
            heads, labels = arc_eager.extract_parse(state)
            parses.add(tuple(heads))
        else:
            for action in arc_eager.allowed(state):
                state_copy = deepcopy(state)
                arc_eager.perform(state_copy, action)
                stack.append(state_copy)

    return parses

class MockSentence:
    def __init__(self, num_tokens):
        self.adjacency = np.zeros((num_tokens, num_tokens), dtype=bool)


class TestArcEager(TestCase):

    def test_dynamic_oracle_is_complete(self):
        SIZE = 4
        arc_eager = ArcEager(1)
        dyn_oracle = ArcEagerDynamicOracle()

        valid_parses = generate_all_projective_parses(SIZE)
        for valid_parse in valid_parses:
            sent = MockSentence(len(valid_parse) + 1)
            for v, u in enumerate(valid_parse):
                sent.adjacency[u, v] = True

            state = arc_eager.state(SIZE)

            while not arc_eager.is_final(state):
                allowed_actions = arc_eager.allowed(state)
                costs = dyn_oracle(state, sent, allowed_actions)
                self.assertEqual(costs.min(), 0)
                index = costs.argmin()
                arc_eager.perform(state, allowed_actions[index])

            heads, labels = arc_eager.extract_parse(state)
            self.assertEqual(tuple(heads), valid_parse)