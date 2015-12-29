import numpy as np


class Action:
    def __str__(self):
        return type(self).__name__

    def num_edges_lost(self, conf, R_gold):
        return sum(R_gold[i, j] for i, j in self.edges_lost(conf))


class Shift(Action):
    def is_valid(self, conf):
        return len(conf.buffer) > 1

    def perform(self, conf):
        conf.stack.append(conf.buffer[0])
        conf.buffer = conf.buffer[1:]

    def edges_lost(self, conf):
        n0 = conf.buffer[0]

        # All edges where N0 is the head and the dependent is in the stack
        lost = [(n0, d) for d in conf.stack]

        # All edges where N0 is dependent and the head is in the stack
        lost += [(h, n0) for h in conf.stack]

        return lost

    def num_edges_lost(self, conf, R_gold):
        n0 = conf.buffer[0]

        # All edges where N0 is the head and the dependent is in the stack
        loss = R_gold[n0, conf.stack].sum()

        # All edges where N0 is dependent and the head is in the stack
        loss += R_gold[conf.stack, n0].sum()

        return loss


class Reduce(Action):
    def is_valid(self, conf):
        # S0 must have a head
        return len(conf.stack) >= 1 and conf.heads[conf.stack[-1]] is not None

    def perform(self, conf):
        conf.stack.pop()

    def edges_lost(self, conf):
        s0 = conf.stack[-1]
        # All edges where S0 is the head and the dependent is in the buffer
        lost = [(s0, d) for d in conf.buffer[:-1]]

        # All edges where S0 is dependent of a node in the buffer
        lost += [(h, s0) for h in conf.buffer[:-1]]

        return lost

    def num_edges_lost(self, conf, R_gold):
        s0 = conf.stack[-1]

        # All edges where S0 is the head and the dependent is in the buffer
        loss = R_gold[s0, conf.buffer[:-1]].sum()

        # All edges where S0 is dependent of a node in the buffer
        loss += R_gold[conf.buffer[:-1], s0].sum()

        return loss


class LeftArc(Action):
    def is_valid(self, conf):
        # S0 must not have a head
        return len(conf.stack) >= 1 and conf.heads[conf.stack[-1]] is None

    def perform(self, conf):
        conf.heads[conf.stack[-1]] = conf.buffer[0]
        conf.stack.pop()

    def edges_lost(self, conf):
        s0 = conf.stack[-1]
        # All edges with S0 as dependent and a head in the buffer N1...
        lost = [(h, s0) for h in conf.buffer[1:]]

        # All edges with S0 as head and a dependent in the buffer N0...
        lost += [(s0, d) for d in conf.buffer]

        return lost

    def num_edges_lost(self, conf, R_gold):
        s0 = conf.stack[-1]
        # All edges with S0 as dependent and a head in the buffer N1...
        loss = R_gold[conf.buffer[1:], s0].sum()

        # All edges with S0 as head and a dependent in the buffer N0...
        loss += R_gold[s0, conf.buffer].sum()

        return loss


class RightArc(Action):
    def is_valid(self, conf):
        # S0 must exists and we should have strictly more than one node in the buffer
        return len(conf.stack) >= 1 and len(conf.buffer) > 1

    def perform(self, conf):
        conf.heads[conf.buffer[0]] = conf.stack[-1]
        conf.stack.append(conf.buffer[0])
        conf.buffer = conf.buffer[1:]

    def edges_lost(self, conf):
        n0 = conf.buffer[0]

        # All edges with N0 as a dependent and the head in S1...
        lost = [(h, n0) for h in conf.stack[:-1]]

        # All edges with N0 as head and a dependent in S
        lost += [(n0, d) for d in conf.stack]

        # All other edges with N0 as a dependent
        lost += [(h, n0) for h in conf.buffer]

        return lost

    def num_edges_lost(self, conf, R_gold):
        n0 = conf.buffer[0]

        # All edges with N0 as a dependent and the head in S1...
        loss = R_gold[conf.stack[:-1], n0].sum()

        # All edges with N0 as head and a dependent in S
        loss += R_gold[n0, conf.stack].sum()

        # All other edges with N0 as a dependent
        loss += R_gold[conf.buffer, n0].sum()

        return loss


ARC_EAGER = [Shift(), Reduce(), LeftArc(), RightArc()]


class ArcEagerParseState:
    def __init__(self, num_tokens, tokens=None, reachability=False):
        n = num_tokens
        self.n = n
        self.stack = []
        self.buffer = list(range(n))
        self.heads = [None for _ in range(n)]
        self.labels = [0 for _ in range(n)]
        self.history = []

        self.actions = ARC_EAGER
        self.tokens = None

        if reachability:
            self.R = np.ones((n, n), dtype=bool)
            self.R[np.diag_indices(n)] = False
            self.R[:, -1] = False

    def _buffer_and_stack(self):
        if self.tokens is None:
            self.tokens = [str(i) for i in range(self.n)]
        stack_nodes = [self.tokens[i] for i in self.stack]
        return "{}]{}".format("".join(stack_nodes), self.tokens[self.buffer[0]])

    def __str__(self):
        return self._buffer_and_stack()

    def full_state(self):
        return self._buffer_and_stack() + " " + " ".join(map(str, self.heads))

    def reachability(self, action):
        R_copy = self.R.copy()
        for i, j in action.edges_lost(self):
            if not self.heads[j] == i:
                R_copy[i, j] = False
        return R_copy

    def valid_actions(self):
        return [action for action in self.actions
                if action.is_valid(self)]

    def __eq__(self, other):
        return tuple(self.stack) == tuple(other.stack) and \
               tuple(self.buffer) == tuple(other.buffer) and \
               tuple(self.heads) == tuple(other.heads)

    def __hash__(self):
        return hash(tuple(self.stack)) + hash(tuple(self.buffer)) + hash(tuple(self.heads))


class ArcEagerDynamicOracle:
    is_optimal = True

    def __call__(self, state, sent, allowed):
        costs = [ARC_EAGER[i].num_edges_lost(state, sent.adjacency) for i in allowed]
        return np.array(costs)


class ArcEager:
    def __init__(self, num_labels):
        self.num_labels = num_labels

    def num_actions(self):
        return len(ARC_EAGER)

    def state(self, num_tokens):
        return ArcEagerParseState(num_tokens)

    def is_final(self, state):
        return len(state.stack) == 0 and len(state.buffer) == 1

    def allowed(self, state):
        # Take labels int account
        return [i for i, action in enumerate(ARC_EAGER)
                if action.is_valid(state)]

    def reference_policy(self):
        return ArcEagerDynamicOracle()

    def action_name(self, action_index):
        return ARC_EAGER[action_index].__class__.__name__

    def describe_action(self, state, action_index):
        return self.action_name(action_index) + "()"

    def perform(self, state, action_index):
        ARC_EAGER[action_index].perform(state)
        state.history.append(action_index)
