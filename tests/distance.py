import unittest

import numpy as np

from src.utils.distance import *


class TestDistance(unittest.TestCase):
    def setUp(self):
        self.A = np.random.rand(10, 3)
        self.B = np.random.rand(10, 3)

    def test_euclidean(self):
        self.assertEqual(euclidean(self.A, self.B).shape, (10,))
        self.assertTrue(np.allclose(euclidean(self.A, self.B), euclidean(self.B, self.A)))
        self.assertTrue(np.allclose(euclidean(self.A, self.B), np.linalg.norm(self.B - self.A, axis=1)))

    def test_manhattan(self):
        self.assertEqual(manhattan(self.A, self.B).shape, (10,))
        self.assertTrue(np.allclose(manhattan(self.A, self.B), manhattan(self.B, self.A)))
        self.assertTrue(np.allclose(manhattan(self.A, self.B), np.sum(np.abs(self.B - self.A), axis=1)))

    def test_chebyshev(self):
        self.assertEqual(chebyshev(self.A, self.B).shape, (10,))
        self.assertTrue(np.allclose(chebyshev(self.A, self.B), chebyshev(self.B, self.A)))
        self.assertTrue(np.allclose(chebyshev(self.A, self.B), np.max(np.abs(self.B - self.A), axis=1)))

    def test_minkowski(self):
        self.assertEqual(minkowski(self.A, self.B).shape, (10,))
        self.assertTrue(np.allclose(minkowski(self.A, self.B), minkowski(self.B, self.A)))
        self.assertTrue(
            np.allclose(minkowski(self.A, self.B), np.power(np.linalg.norm(self.B - self.A, ord=2, axis=1), 0.5)))
        self.assertTrue(
            np.allclose(minkowski(self.A, self.B, p=2.5),
                        np.power(np.linalg.norm(self.B - self.A, ord=2.5, axis=1), 0.4)))
        self.assertTrue(
            np.allclose(minkowski(self.A, self.B, p=0.5),
                        np.power(np.linalg.norm(self.B - self.A, ord=0.5, axis=1), 2)))

    def test_cosine(self):
        self.assertEqual(cosine(self.A, self.B).shape, (10,))
        self.assertTrue(np.allclose(cosine(self.A, self.B), cosine(self.B, self.A)))
        self.assertTrue(np.allclose(cosine(self.A, self.B),
                                    np.linalg.norm(self.B - self.A, axis=1) / (
                                            np.linalg.norm(self.A, axis=1) * np.linalg.norm(self.B, axis=1))))
