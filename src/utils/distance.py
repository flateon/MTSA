import numpy as np

from src.utils.decomposition import get_decomposition


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
    def __init__(self, period=24, weight=None, decomposition='moving_average', distance=chebyshev):
        self.period = period
        self.decompose, n_components = get_decomposition(decomposition)
        self.weight = weight if weight is not None else np.ones(n_components) / n_components
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
        a_decomposed = self.decompose(a, self.period)
        if self.b_decomposed is None or self.b_id != id(b):
            self.b_id = id(b)
            self.b_decomposed = self.decompose(b.reshape(b.shape[0], b.shape[1], -1), self.period)

        dist = sum((self.distance(a_d, b_d) * w for a_d, b_d, w in zip(a_decomposed, self.b_decomposed, self.weight)))
        return dist
