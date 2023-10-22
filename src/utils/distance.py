import numpy as np

from src.utils.decompose import seasonal_decompose


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


def zero(a, b):
    return np.zeros(len(b))


class Decompose:
    def __init__(self, period=24, weight=(0.1, 0.5, 0.4), distance=euclidean):
        self.period = period
        self.weight = weight
        self.distance = distance
        self.b_decomposed = None
        self.b_id = None

    def __call__(self, a, b):
        a_t, a_s, a_r = seasonal_decompose(a, self.period)
        if self.b_decomposed is None or self.b_id != id(b):
            self.b_decomposed = seasonal_decompose(b, self.period)
            self.b_id = id(b)
        b_t, b_s, b_r = self.b_decomposed

        return self.weight[0] * self.distance(a_t, b_t) + self.weight[1] * self.distance(a_s, b_s) + self.weight[
            2] * self.distance(a_r, b_r)
        # return self.distance(a_t, b_t)
        # return self.distance(a_s, b_s)
        # return self.distance(a_r, b_r)
