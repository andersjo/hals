from enum import IntEnum

import numpy as np
import tensorflow as tf

from sentences import NA_FEATURE_INDEX


class TokenRole(IntEnum):
    S0 = 0
    S1 = 1
    S2 = 2
    S3 = 3
    N0 = 4
    N1 = 5
    N2 = 6

    S0_PARENT = 7
    S0_GRANDPARENT = 8

    S0_CHILD_LEFT = 9
    S0_CHILD_LEFT_2 = 10
    S0_CHILD_RIGHT = 11
    S0_CHILD_RIGHT_2 = 12

    N0_CHILD_LEFT = 13
    N0_CHILD_LEFT_2 = 14


# TODO What features can be extracted really is specific to each transition system
class TokeRoleAssignment:
    def __init__(self):
        self.indices = np.ones(len(TokenRole), dtype=int) * -1

    def update(self, state):
        indices = self.indices
        indices.fill(NA_FEATURE_INDEX)

        # S0
        if len(state.stack) > 0:
            s0 = state.stack[-1]
            indices[TokenRole.S0] = s0
            s0_parent = state.heads[s0]
            if s0_parent is not None:
                indices[TokenRole.S0_PARENT] = s0_parent
                if state.heads[s0_parent] is not None:
                    indices[TokenRole.S0_GRANDPARENT] = state.heads[s0_parent]

            children_of_s0 = [v for v, u in enumerate(state.heads) if u == s0]
            if len(children_of_s0) >= 1:
                indices[TokenRole.S0_CHILD_LEFT] = children_of_s0[0]
                indices[TokenRole.S0_CHILD_RIGHT] = children_of_s0[-1]
            if len(children_of_s0) >= 2:
                indices[TokenRole.S0_CHILD_LEFT_2] = children_of_s0[1]
                indices[TokenRole.S0_CHILD_RIGHT_2] = children_of_s0[-2]

        if len(state.stack) >= 2:
            indices[TokenRole.S1] = state.stack[-2]
        if len(state.stack) >= 3:
            indices[TokenRole.S2] = state.stack[-3]

        # N0
        n0 = state.buffer[0]
        indices[TokenRole.N0] = n0

        children_of_n0 = [v for v, u in enumerate(state.heads) if u == n0]
        if len(children_of_n0) >= 1:
            indices[TokenRole.N0_CHILD_LEFT] = children_of_n0[0]
        if len(children_of_n0) >= 2:
            indices[TokenRole.N0_CHILD_LEFT_2] = children_of_n0[1]

        if len(state.buffer) >= 2:
            indices[TokenRole.N1] = state.buffer[1]
        if len(state.buffer) >= 3:
            indices[TokenRole.N2] = state.buffer[2]

    def __getitem__(self, item):
        return self.indices[item]


class InputBuilder:
    def __init__(self, token_role, namespace):
        self.token_role = token_role
        self.namespace = namespace
        self.features = []
        self.name = "{}_{}".format(namespace, token_role.name)

    def reset(self):
        self.features.clear()


class EmbeddingBuilder(InputBuilder):
    pass


class FeatureTemplate:
    def __init__(self, builders):
        self.builders = builders
        self.role_assignments = TokeRoleAssignment()

    def update(self, state, sent):
        self.role_assignments.update(state)

        for builder in self.builders:
            builder.reset()
            token_index = self.role_assignments[builder.token_role]
            if token_index != -1:
                token = sent.tokens[token_index]
                feature_indices = token.features.get(builder.namespace)
                if feature_indices:
                    builder.features.extend(feature_indices)

            if not len(builder.features):
                builder.features.append((NA_FEATURE_INDEX, 1.0))


class LearnerModel:
    def __init__(self):
        self.fillers = []
        self.num_classes = None

        # Placeholders
        self.gold_ph = None
        self.dropout_keep_p_ph = None

        # Graph nodes
        self.loss = None
        self.logits = None
        self.pred = None

        self._token_role_assignment = TokeRoleAssignment()

    def prepare_feed(self, sent, state):
        # TODO `LearnerModel` should have no state
        self._token_role_assignment.update(state)
        role_assignment = self._token_role_assignment.indices

        feed_dict = {}
        for filler in self.fillers:
            filler.update_feed([(sent, role_assignment)], feed_dict)

        return feed_dict

    def prepare_feed_batch(self, sentences, states):
        role_assignment_list = []
        for state in states:
            self._token_role_assignment.update(state)
            role_assignment_list.append(self._token_role_assignment.indices.copy())

        feed_dict = {}
        for filler in self.fillers:
            filler.update_feed(list(zip(sentences, role_assignment_list)), feed_dict)

        return feed_dict






        pass


class NsPlaceholderFiller:
    """
    This class exposes three public placeholders:

       `input_ph` and `weights_ph`, which are both [batch_size, num_features, num_roles] tensors.
       `lengths_ph`, a [batch_size] tensor with the actual lengths for the current input.
    """

    def __init__(self, ns, ns_max_len, token_roles):
        self.namespace = ns
        self.token_roles = token_roles
        self.ns_max_len = ns_max_len

        # Placeholders
        self.input_ph = tf.placeholder(tf.int32, [None, ns_max_len, len(token_roles)],
                                       name='input_ph_ns_' + ns)
        self.weights_ph = tf.placeholder(tf.float32, [None, ns_max_len, len(token_roles)],
                                         name='weights_ph_ns_' + ns)
        self.lengths_ph = tf.placeholder(tf.int32, [None, 1],
                                         name='lengths_ph_ns_' + ns)

    def update_feed(self, sent_and_role_assignments, feed_dict):
        batch_size = len(sent_and_role_assignments)

        # Allocate feed values
        input_ = np.zeros([batch_size, self.ns_max_len, len(self.token_roles)], dtype=np.int32)
        weights = np.ones_like(input_, dtype=np.float32)
        lengths = np.zeros([batch_size, 1])

        # Fill the feed values based on current assignments
        for instance_i in range(batch_size):
            sent, role_assignments = sent_and_role_assignments[instance_i]
            tokens = sent.tokens
            for role_i, token_role in enumerate(self.token_roles):
                token_index = role_assignments[token_role]
                if token_index == -1:
                    continue

                token_features = tokens[token_index].features.get(self.namespace)
                if not token_features:
                    continue

                for feat_i in range(len(token_features)):
                    feat_id, feat_weight = token_features[feat_i]
                    input_[instance_i, feat_i, role_i] = feat_id
                    weights[instance_i, feat_i, role_i] = feat_weight

                lengths[instance_i, 0] = len(token_features)

        # Map placeholders to values
        feed_dict[self.input_ph] = input_
        feed_dict[self.weights_ph] = weights
        feed_dict[self.lengths_ph] = lengths


def embed_ns(ph_filler, embeds, name, reduce_fn=tf.reduce_mean, scale=True):
    with tf.name_scope('embed_' + name):
        embedded_feats = tf.gather(embeds, ph_filler.input_ph)

        if scale:
            scaling = tf.expand_dims(ph_filler.weights_ph, -1)
            embedded_feats = tf.mul(embedded_feats, scaling)

        reduced_by_feat = reduce_fn(embedded_feats, 1)

        # TODO this cannot be the best way to flatten a tensor to [batch_size, -1] ?
        batch_size_node = tf.reshape(tf.shape(reduced_by_feat)[0], [1, 1])
        shape_node = tf.concat(0, [batch_size_node, tf.reshape(tf.constant(-1), [1, 1])])
        shape_node = tf.squeeze(shape_node)

        num_units = embeds.get_shape()[1] * ph_filler.input_ph.get_shape()[2]

        return tf.reshape(reduced_by_feat, shape_node), num_units.value


def make_zhang_simplified_input(ns_embeddings):
    roles = [TokenRole.S0, TokenRole.S1, TokenRole.S2,
             TokenRole.S0_PARENT, TokenRole.S0_GRANDPARENT,
             TokenRole.S0_CHILD_RIGHT, TokenRole.S0_CHILD_RIGHT_2,
             TokenRole.S0_CHILD_LEFT, TokenRole.N0_CHILD_LEFT_2,
             TokenRole.N0, TokenRole.N1, TokenRole.N2,
             TokenRole.N0_CHILD_LEFT, TokenRole.N0_CHILD_LEFT_2]

    fillers = []
    inputs = []
    num_units = 0
    for ns in ns_embeddings.keys():
        filler = NsPlaceholderFiller(ns, 1, roles)
        fillers.append(filler)

        input_for_ns, num_units_for_ns = embed_ns(filler, ns_embeddings[ns], 'ns_' + ns, scale=True)
        inputs.append(input_for_ns)
        num_units += num_units_for_ns

    return tf.concat(1, inputs), fillers, num_units


def make_hidden(input_layer, num_inputs, num_hidden) -> tf.Tensor:
    with tf.name_scope('hidden'):
        w = tf.Variable(tf.truncated_normal([num_inputs, num_hidden], stddev=0.1), name='w')
        b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[num_hidden]), name='b')

        act = tf.nn.relu(tf.nn.xw_plus_b(input_layer, w, b), name='act')
        return act


def make_output_and_loss(last_hidden, num_inputs, num_classes, gold_ph) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.name_scope('output'):
        w = tf.Variable(tf.truncated_normal([num_inputs, num_classes], stddev=0.1), name='w')
        tf.histogram_summary("output weights", w)

        b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[num_classes]), name='b')
        tf.histogram_summary("output biases", b)

        logits = tf.nn.xw_plus_b(last_hidden, w, b, name='act')
        tf.histogram_summary("output logits", w)

    with tf.name_scope('loss'):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits, gold_ph)
        loss = tf.reduce_sum(losses)
        tf.scalar_summary("sigmoid cross entropy", loss)

    with tf.name_scope('pred'):
        pred = tf.arg_max(logits, 1)


    return loss, pred, logits


def make_zhang_simplified_model(feat_embeddings_by_ns, num_classes, num_hidden) -> LearnerModel:
    """
    The inputs of this model is based on the Zhang and Manning (2014) NN feature model.

    Some adaptations were necessary to account for the fact that the Zhang and Manning model is based on ArcStandard,
    while this model targets ArcEager.

    :param feat_embeddings_by_ns:
    :param num_classes:
    :param num_hidden:
    :return:
    """
    input_node, fillers, num_units = make_zhang_simplified_input(feat_embeddings_by_ns)
    last_hidden_node = make_hidden(input_node, num_units, num_hidden)

    # Apply dropout to last layer
    dropout_keep_p_ph = tf.placeholder(tf.float32, shape=[], name='dropout_keep_p')
    last_hidden_node = tf.nn.dropout(last_hidden_node, dropout_keep_p_ph)


    # TODO consider how to model more structured outputs, taking groups into account
    gold_ph = tf.placeholder(tf.float32, [None, num_classes], name='gold_ph')
    loss, pred, logits = make_output_and_loss(last_hidden_node, num_hidden, num_classes, gold_ph)

    # Update model
    model = LearnerModel()
    model.num_classes = num_classes
    model.fillers = fillers
    model.gold_ph = gold_ph
    model.dropout_keep_p_ph = dropout_keep_p_ph
    model.loss = loss
    model.pred = pred
    model.logits = logits

    return model


def init_embeddings(corpus, ns_embedding_sizes, pretrained):
    feat_embeddings = {}
    for ns, feat_dict in corpus.ns_feature_dicts.items():
        M_init = np.random.uniform(-1, 1, [len(feat_dict), ns_embedding_sizes[ns]]).astype(np.float32)
        if ns in pretrained:
            feat_names = [feat_dict.id_to_name[i] for i in range(len(feat_dict))]
            M_init = pretrained[ns].ix[feat_names].values.astype(np.float32)
            missing_mask = np.isnan(M_init)
            M_init[missing_mask] = np.random.uniform(-1, 1, missing_mask.sum())

        feat_embeddings[ns] = tf.Variable(M_init)

    return feat_embeddings
