import unittest

import numpy as np

from src.utils.distance import *


class TestDistance(unittest.TestCase):
    def setUp(self):
        self.num_samples = 10
        self.A = [np.random.rand(self.num_samples, 32, 3),
                  np.random.rand(self.num_samples, 32)]
        self.B = [np.random.rand(self.num_samples, 32, 3),
                  np.random.rand(self.num_samples, 32)]

    def test_euclidean(self):
        for a, b in zip(self.A, self.B):
            self.assertEqual(euclidean(a, b).shape, (self.num_samples,))
            self.assertTrue(np.allclose(euclidean(a, b), euclidean(b, a)))

    def test_manhattan(self):
        for a, b in zip(self.A, self.B):
            self.assertEqual(manhattan(a, b).shape, (self.num_samples,))
            self.assertTrue(np.allclose(manhattan(a, b), manhattan(b, a)))

    def test_chebyshev(self):
        for a, b in zip(self.A, self.B):
            self.assertEqual(chebyshev(a, b).shape, (self.num_samples,))
            self.assertTrue(np.allclose(chebyshev(a, b), chebyshev(b, a)))

    def test_minkowski(self):
        for a, b in zip(self.A, self.B):
            self.assertEqual(minkowski(a, b).shape, (self.num_samples,))
            self.assertTrue(np.allclose(minkowski(a, b), minkowski(b, a)))

    def test_cosine(self):
        for a, b in zip(self.A, self.B):
            self.assertEqual(cosine(a, b).shape, (self.num_samples,))
            self.assertTrue(np.allclose(cosine(a, b), cosine(b, a)))

    def test_zero(self):
        for a, b in zip(self.A, self.B):
            self.assertEqual(zero(a, b).shape, (self.num_samples,))
            self.assertTrue(np.allclose(zero(a, b), zero(b, a)))
            self.assertTrue(np.allclose(zero(a, b), np.zeros(10)))

    def test_distance_calculation(self):
        for a, b in zip(self.A, self.B):
            decompose = DecomposeDistance(period=5, weight=(0.5, 0.5), distance=euclidean)

            self.assertEqual(decompose(a, b).shape, (self.num_samples,))
            self.assertTrue(np.allclose(decompose(a, b), decompose(b, a)))
