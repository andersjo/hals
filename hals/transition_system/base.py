class AbstractTransitionSystem:
    def __init__(self, num_labels):
        self.num_labels = num_labels

    def num_actions(self):
        raise NotImplementedError()

    def state(self, num_tokens):
        raise NotImplementedError()

    def is_final(self, state):
        raise NotImplementedError()

    def extract_parse(self, state):
        raise NotImplementedError()

    def allowed(self, state):
        raise NotImplementedError()

    def reference_policy(self):
        raise NotImplementedError()

    def action_name(self, action_index):
        return 'Action=' + str(action_index)

    def describe_action(self, state, action_index):
        raise self.action_name(action_index) + ' at ' + str(state)

    def perform(self, state, action_index):
        raise NotImplementedError()


class AbstractReferencePolicy:

    def is_optimal(self):
        raise NotImplementedError()