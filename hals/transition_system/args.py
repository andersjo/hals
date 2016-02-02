from transition_system import TRANSITION_SYSTEMS, ArcEager

name_to_class = {cls.__name__: cls for cls in TRANSITION_SYSTEMS}

def add_train_args(arg_parser):
    arg_parser.add_argument('--transition-system',
                            choices=name_to_class.keys(), default='ArcEager',
                            )

    # Add options specific to each transition system
    for cls in TRANSITION_SYSTEMS:
        pass

def make_from_args(args):
    cls = name_to_class[args.transition_system]
    return cls