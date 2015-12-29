import logging
import re
from collections import defaultdict

import numpy as np

# Features that are out-of-vocabulary going from train to test
OOV_FEATURE = '*OOV*'
OOV_FEATURE_INDEX = 0
# Features that are undefined for other reasons
NA_FEATURE = '*NA*'
NA_FEATURE_INDEX = 1
DEFAULT_NS = '^'


class FeatureDict:
    def __init__(self):
        self.id_to_name = {}
        self.name_to_id = {OOV_FEATURE: OOV_FEATURE_INDEX,
                           NA_FEATURE: NA_FEATURE_INDEX}

    def map(self, name, frozen=False):
        feat_id = self.name_to_id.get(name)
        if feat_id is None and not frozen:
            feat_id = len(self.name_to_id)
            self.name_to_id[name] = feat_id

        return feat_id

    def update_reverse_mapping(self):
        self.id_to_name = {feat_id: feat_name for feat_name, feat_id in self.name_to_id.items()}

    def __len__(self):
        return len(self.name_to_id)


class Corpus:
    def __init__(self):
        self.ns_feature_dicts = defaultdict(FeatureDict)
        self.frozen = False

    def add_dataset(self, name, instances, frozen=False):
        for instance in instances:
            for token in instance.tokens:
                token.features = self._map_features(token)

    def update_reverse_mapping(self):
        for feat_dict in self.ns_feature_dicts.values():
            feat_dict.update_reverse_mapping()

    def freeze(self):
        self.frozen = True

    def _map_features(self, token):
        features_by_ns = {}
        for ns, feature_list in token.str_features.items():
            feat_dict = self.ns_feature_dicts[ns]
            features = []
            for feat_name, feat_val in feature_list:
                feat_id = feat_dict.map(feat_name, frozen=self.frozen)
                if feat_id is None:
                    feat_id = OOV_FEATURE_INDEX

                features.append((feat_id, feat_val))

            features_by_ns[ns] = features

        return features_by_ns


class Sentence:
    def __init__(self):
        self.tokens = []

    def has_edge(self, u, v):
        return self.tokens[v].head == u

    def __len__(self):
        return len(self.tokens)

    def num_correct(self, heads, labels):
        # FIXME should the root token appear in the `heads` and `labels`?
        num_labeled = 0
        num_unlabeled = 0
        for v, u in enumerate(heads):
            dep = self.tokens[v]
            if dep.head == u:
                num_unlabeled += 1
                if dep.label == labels[v]:
                    num_labeled += 1

        return num_unlabeled, num_labeled

    def finish(self):
        # Add artificial root node
        root_token = Token()
        root_token.head = -1
        self.tokens.append(root_token)

        self.adjacency = np.zeros((len(self.tokens), len(self.tokens)), dtype=bool)
        for j, token in enumerate(self.tokens[:-1]):
            self.adjacency[token.head, j] = True


class Token:
    def __init__(self):
        self.str_features = {}


class SentenceParser:
    def __init__(self, filename):
        self.filename = filename
        self.reset()

    def process(self, line, line_no):
        self.line_no = line_no

        parts = line.split("|", maxsplit=1)
        if len(parts) != 2:
            raise self.parse_error("Missing bar |")
        self._token = Token()
        self._parse_header(parts[0])
        self._parse_features("|" + parts[1])
        self.sent.tokens.append(self._token)

    def _parse_header(self, header_str):
        match = re.search("(-?\d+)-(\S+) '(.*)", header_str)
        if not match:
            raise self._parse_error("Invalid header (part before first |)")

        groups = match.groups()
        self._token.head = int(groups[0])
        self._token.label = groups[1]
        self._token.id = groups[2]

    def _parse_features(self, feature_str):
        ns_name = DEFAULT_NS
        for part in re.split(r"\s+", feature_str):
            if not len(part):
                continue

            if part[0] == '|':
                # Namespace declaration
                ns_name = part[1:] or DEFAULT_NS
            else:
                # Feature, with an optional value
                last_col_pos = part.rfind(':')
                if last_col_pos > 0:
                    feature_name = part[:last_col_pos]
                    try:
                        feature_val = float(part[last_col_pos + 1:])
                        # noinspection PyBroadException
                    except Exception as e:
                        logging.info(
                            "Failed to interpret '{}' as a float. Defaulting to 1.0 ".format(part[last_col_pos + 1:]))
                        feature_val = 1.0
                else:
                    feature_name = part
                    feature_val = 1.0

                if ns_name not in self._token.str_features:
                    self._token.str_features[ns_name] = []
                self._token.str_features[ns_name].append((feature_name, feature_val))

    def reset(self):
        self.sent = Sentence()

    def _parse_error(self, message):
        return RuntimeError("File input error: {} in {} at line {}".format(message,
                                                                           self.filename, self.line_no))


def read_sentences(filename):
    sentence_parser = SentenceParser(filename)
    sentences = []
    for line_no, line in enumerate(open(filename), 1):
        if line == "\n":
            sentences.append(sentence_parser.sent)
            sentences[-1].finish()
            sentence_parser.reset()
        else:
            sentence_parser.process(line, line_no)

    return sentences


if __name__ == '__main__':
    sentences = list(read_sentences("/Users/anders/data/treebanks/hanstholm/en.gweb_weblogs.test.gold_fine.hanstholm"))
    corpus = Corpus()
    corpus.add_dataset('train', sentences)
    # corpus.add_unknown_features()
    corpus.freeze()
