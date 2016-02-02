import argparse
from pathlib import Path

import transition_system
from cli.util import type_with_ns
from cli.train import train
from cli.predict import predict

parser = argparse.ArgumentParser(description="Hals, a transition-based dependency parser")
subparsers = parser.add_subparsers()

# Training
train_parser = subparsers.add_parser('train', help="Train a model")
train_parser.add_argument('--builder', help="Model builder")
train_parser.add_argument('--train', help="Training set", type=Path)
train_parser.add_argument('--dev', help="Development set", type=Path)
train_parser.add_argument('--ns-init', help="Initialize the embeddings for features in a namespace from this HDF-file."
                                      "Specify as 'ns:filename'. Can occur multiple times.",
                          type=type_with_ns(Path), action='append', default=[])
train_parser.add_argument('--ns-dim', help="Size of embedding for the given namespace. Format is 'ns:dim'. "
                                     "Can occur multiple times.",
                    action='append', default=[], type=type_with_ns(int))
train_parser.set_defaults(fn=train)

transition_system_group = train_parser.add_argument_group('Transition system')
transition_system.add_train_args(transition_system_group)

model_group = train_parser.add_argument_group('Model builder')
model_group.add_argument('--ns-default-token-role')
model_group.add_argument('--ns-token-role')
model_group.add_argument('--role-set', help="Predefined role set.")
model_group.add_argument('--ns-role-set', help="Override roles for a given namespace.")


learner_group = train_parser.add_argument_group('Learner')
learner_group.add_argument('--early-stop-after',
                           help="Terminate after this number of epochs with no increase in performance on dev set.",
                           type=int, default=3)
learner_group.add_argument('--clip-gradient', action='store_true')

predict_parser = subparsers.add_parser('predict', help="Use a trained model to make predictions")
predict_parser.set_defaults(fn=predict)

args = parser.parse_args()
args.fn(args)

