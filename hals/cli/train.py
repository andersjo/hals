from collections import defaultdict

import pandas as pd

import transition_system
from feature import init_embeddings, make_zhang_simplified_model
from fnn_learner import FnnLearner
from learner import RandomLearner
from sentences import Corpus, read_sentences
from transition_parser.threaded_parser import ThreadedTransitionParser


def train(args):
    num_labels = 1

    # Read data
    train_sents = list(read_sentences(str(args.train)))
    dev_sents = list(read_sentences(str(args.dev)))

    # TODO Better name? `Corpus` is really feature mappings
    corpus = Corpus()
    corpus.add_dataset('train', train_sents)
    corpus.freeze()
    corpus.add_dataset('dev', dev_sents)
    corpus.update_reverse_mapping()

    # Initialize embeddings
    # Some default values for namespaces
    ns_embedding_sizes = defaultdict(lambda: 50)
    ns_embedding_sizes.update({'w': 100, 'p': 12})

    pretrained = {}
    for ns_init in args.ns_init:
        ns, filename = ns_init.split(":")
        pretrained[ns] = pd.read_hdf(filename)
        ns_embedding_sizes[ns] = pretrained[ns].shape[1]

    for ns_dim in args.ns_dim:
        ns, dim = ns_dim
        if ns in pretrained:
            assert pretrained[ns].shape[1] != dim, "Incompatible embedding sizes from 'ns-init' and 'ns-dim'"
        ns_embedding_sizes[ns] = dim

    feat_embeddings = init_embeddings(corpus, ns_embedding_sizes, pretrained)

    # Transition system
    t_sys_cls = transition_system.make_from_args(args)
    t_sys = t_sys_cls(num_labels)

    # Build tensorflow model for learner
    tf_model = make_zhang_simplified_model(feat_embeddings, t_sys.num_actions(), num_hidden=100)

    # Initialize parser
    # random_learner = RandomLearner(t_sys.num_actions())
    fnn_learner = FnnLearner(tf_model,
                             clip_gradient=args.clip_gradient)
    # learner = random_learner
    learner = fnn_learner

    parser = ThreadedTransitionParser(t_sys, learner, early_stop_after=args.early_stop_after)
    parser.fit(train_sents, dev_sents)
