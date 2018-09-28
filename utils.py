import os
import pickle
import sys
import re
import string
import numpy as np

RE_WS_PRE_PUCT = re.compile(u'\s+([^a-zA-Z\d])')
RE_WIKI_ENT = re.compile(r'.*wiki\/(.*)')
RE_WS = re.compile('\s+')


def pickle_load(path):
    assert os.path.exists(path)

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def normalise_form(sf):
    sf = sf.lower()
    sf = RE_WS_PRE_PUCT.sub(r'\1', sf)
    sf = RE_WS.sub(' ', sf)
    return sf


def iter_derived_forms(sf):
    yield sf
    yield sf.replace("'s", "")
    yield ''.join(c for c in sf if not c in string.punctuation)

    if sf.startswith('The') or sf.startswith('the'):
        yield sf[4:]

    comma_parts = sf.split(',')[:-1]
    for i in range(len(comma_parts)):
        yield ''.join(comma_parts[:i + 1])
    if comma_parts:
        yield ''.join(comma_parts)

    colon_idx = sf.find(':')
    if colon_idx != -1:
        yield sf[:colon_idx]


def normalize(v):
    if len(v.shape) == 1:
        return v / (np.linalg.norm(v) + 10**-11)
    elif len(v.shape) == 2:
        norm = np.linalg.norm(v, axis=1) + 10**-11
        return v / norm[:, None]
    else:
        print("normalize only accepts arrays of dimensions 1 or 2.")
        sys.exit(1)


def get_normalised_forms(sf):
    return set(normalise_form(f) for f in iter_derived_forms(sf))


def reverse_dict(d):

    return {v: k for k, v in d.items()}


def relu(x):
    x_c = x.copy()
    x_c[x_c < 0] = 0
    return x_c


def equalize_len(data, max_size):
    d = data.copy()
    l = len(d)
    if l >= max_size:
        return d[:max_size]
    else:
        for _ in range(max_size - l):
            d.append(0)

        return d


if __name__ == '__main__':
    pass
