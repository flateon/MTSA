import numpy as np

from src.utils.decomposition import moving_average


def euclidean(a, b):
    """
    :param a: (n, d) or (n, d, c)
    :param b: (n, d) or (n, d, c)
    :return: (n,)
    """
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    return np.linalg.norm(b - a, axis=1)


def manhattan(a, b):
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    return np.sum(np.abs(b - a), axis=1)


def chebyshev(a, b):
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    return np.max(np.abs(b - a), axis=1)


def minkowski(a, b, p=2):
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    return np.power(np.linalg.norm(b - a, ord=p, axis=1), 1 / p)


def cosine(a, b):
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    return np.linalg.norm(b - a, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def zero(a, b):
    b = b.reshape(b.shape[0], -1)
    return np.zeros(len(b))


class DecomposeDistance:
    def __init__(self, period=24, weight=(0.5, 0.5), distance=euclidean):
        self.period = period
        self.weight = weight
        self.distance = distance
        self.b_decomposed = None
        self.b_id = None

    def __call__(self, a, b):
        """
        :param a: (n, d) or (n, d, c)
        :param b: (n, d) or (n, d, c)
        :return: (n,)
        """
        a = a.reshape(a.shape[0], a.shape[1], -1)
        a_t, a_s = moving_average(a, self.period)
        if self.b_decomposed is None or self.b_id != id(b):
            self.b_id = id(b)
            b = b.reshape(b.shape[0], b.shape[1], -1)
            self.b_decomposed = moving_average(b, self.period)

        b_t, b_s = self.b_decomposed

        return self.weight[0] * self.distance(a_t, b_t) + self.weight[1] * self.distance(a_s, b_s)
        # return self.distance(a_t, b_t)
        # return self.distance(a_s, b_s)
