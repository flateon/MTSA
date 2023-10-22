import unittest

import numpy as np

from src.utils.distance import *


class TestDistance(unittest.TestCase):
    def setUp(self):
        self.num_samples = 10
        self.A = np.random.rand(self.num_samples, 32)
        self.B = np.random.rand(self.num_samples, 32)

    def test_euclidean(self):
        self.assertEqual(euclidean(self.A, self.B).shape, (self.num_samples,))
        self.assertTrue(np.allclose(euclidean(self.A, self.B), euclidean(self.B, self.A)))
        self.assertTrue(np.allclose(euclidean(self.A, self.B), np.linalg.norm(self.B - self.A, axis=1)))

    def test_manhattan(self):
        self.assertEqual(manhattan(self.A, self.B).shape, (self.num_samples,))
        self.assertTrue(np.allclose(manhattan(self.A, self.B), manhattan(self.B, self.A)))
        self.assertTrue(np.allclose(manhattan(self.A, self.B), np.sum(np.abs(self.B - self.A), axis=1)))

    def test_chebyshev(self):
        self.assertEqual(chebyshev(self.A, self.B).shape, (self.num_samples,))
        self.assertTrue(np.allclose(chebyshev(self.A, self.B), chebyshev(self.B, self.A)))
        self.assertTrue(np.allclose(chebyshev(self.A, self.B), np.max(np.abs(self.B - self.A), axis=1)))

    def test_minkowski(self):
        self.assertEqual(minkowski(self.A, self.B).shape, (self.num_samples,))
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
        self.assertEqual(cosine(self.A, self.B).shape, (self.num_samples,))
        self.assertTrue(np.allclose(cosine(self.A, self.B), cosine(self.B, self.A)))
        self.assertTrue(np.allclose(cosine(self.A, self.B),
                                    np.linalg.norm(self.B - self.A, axis=1) / (
                                            np.linalg.norm(self.A, axis=1) * np.linalg.norm(self.B, axis=1))))

    def test_zero(self):
        self.assertEqual(zero(self.A, self.B).shape, (self.num_samples,))
        self.assertTrue(np.allclose(zero(self.A, self.B), zero(self.B, self.A)))
        self.assertTrue(np.allclose(zero(self.A, self.B), np.zeros(10)))

    def test_distance_calculation(self):
        decompose = DecomposeDistance(period=5, weight=(0.1, 0.5, 0.4), distance=euclidean)

        self.assertEqual(decompose(self.A, self.B).shape, (self.num_samples,))
        self.assertTrue(np.allclose(decompose(self.A, self.B), decompose(self.B, self.A)))
