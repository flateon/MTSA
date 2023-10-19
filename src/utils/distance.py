import numpy as np

import unittest


def euclidean(a, b):
    """
    :param a: (n, d)
    :param b: (n, d)
    :return: (n,)
    """
    return np.linalg.norm(b - a, axis=1)


def manhattan(a, b):
    return np.sum(np.abs(b - a), axis=1)


def chebyshev(a, b):
    return np.max(np.abs(b - a), axis=1)


def minkowski(a, b, p=2):
    return np.power(np.linalg.norm(b - a, ord=p, axis=1), 1 / p)


def cosine(a, b):
    return np.linalg.norm(b - a, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
