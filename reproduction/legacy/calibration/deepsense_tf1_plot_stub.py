import collections
import numpy as np

_since_last_flush = collections.defaultdict(lambda: {})
_iter = [0]


def tick():
    _iter[0] += 1


def plot(name, value):
    _since_last_flush[name][_iter[0]] = value


def flush():
    prints = []
    for name, vals in list(_since_last_flush.items()):
        prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
    print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
    _since_last_flush.clear()
