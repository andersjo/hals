import argparse
from pathlib import Path

import pandas as pd
import time

from arc_eager import ArcEager
from feature import init_embeddings, make_zhang_simplified_model
from fnn_learner import FnnLearner
from learner import RandomLearner
from sentences import Corpus, read_sentences
from transition_parser.threaded_parser import ThreadedTransitionParser

parser = argparse.ArgumentParser(description="Hals, a transition-based dependency parser")
parser.add_argument('--train', help="Training set", type=Path)
parser.add_argument('--dev', help="Development set", type=Path)
parser.add_argument('--ns-init', help="Initialize the embeddings for features in a namespace from this HDF-file."
                                      "Specify as 'ns:filename'. Can occur multiple times.",
                    action='append', default=[])
parser.add_argument('--ns-dim', help="Size of embedding for the given namespace. Format is 'ns:dim'. "
                                     "Can occur multiple times.",
                    action='append', default=[])
args = parser.parse_args()

num_labels = 1

# Read data
train_sents = list(read_sentences(str(args.train)))
dev_sents = list(read_sentences(str(args.train)))
# `Corpus` is really feature mappings
corpus = Corpus()
corpus.add_dataset('train', train_sents)
corpus.freeze()
corpus.add_dataset('dev', dev_sents)
corpus.update_reverse_mapping()

# Initialize embeddings
# Some default values
ns_embedding_sizes = {'w': 100, 'p': 12}
pretrained = {}
for ns_init in args.ns_init:
    ns, filename = ns_init.split(":")
    pretrained[ns] = pd.read_hdf(filename)
    ns_embedding_sizes[ns] = pretrained[ns].shape[1]

for ns_dim in args.ns_dim:
    ns, dim_str = ns_dim.split(":")
    dim = int(dim_str)
    if ns in pretrained:
        assert pretrained[ns].shape[1] != dim, "Incompatible embedding sizes from 'ns-init' and 'ns-dim'"
    ns_embedding_sizes[ns] = dim

feat_embeddings = init_embeddings(corpus, ns_embedding_sizes, pretrained)

# Transition system
arc_eager_trans = ArcEager(num_labels)

# Build tensorflow model for learner
tf_model = make_zhang_simplified_model(feat_embeddings, arc_eager_trans.num_actions(), num_hidden=100)

# Initialize parser
random_learner = RandomLearner(arc_eager_trans.num_actions())
fnn_learner = FnnLearner(tf_model)
# learner = random_learner
learner = fnn_learner

parser = ThreadedTransitionParser(arc_eager_trans, learner)
parser.fit(train_sents, dev_sents)

# many_sents = []
# for sent in dev_sents:
#     for i in range(5):
#         many_sents.append(sent)

# print("Parsing many")
# time_begin = time.perf_counter()
# parser.parse_many(many_sents)
# elapsed = time.perf_counter() - time_begin
# print("Parsing end. Elapsed: ", elapsed)
# print("Sents per sec: {}".format(len(many_sents) / elapsed))
