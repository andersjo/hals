import argparse
from functools import partial


def type_with_ns(embedded_type):
    return partial(type_with_ns_parser, embedded_type=embedded_type)

def type_with_ns_parser(val, embedded_type):
    parts = val.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Argument should be of format 'ns:{}'".format(embedded_type.__name__))
    try:
        val_part = embedded_type(parts[1])
    except Exception:
        raise argparse.ArgumentTypeError("Argument '{}' could not be converted to proper type".format(parts[1]))

    return parts[0], val_part
