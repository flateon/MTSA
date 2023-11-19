import unittest

import numpy as np

from src.utils.decomposition import *


class TestDecomposition(unittest.TestCase):
    def setUp(self):
        self.num_samples = 10
        self.data = [np.random.rand(self.num_samples, 96, 3),
                     np.random.rand(self.num_samples, 96, 1),
                     np.random.rand(self.num_samples, 50, 2),
                     np.random.rand(self.num_samples, 52, 2), ]

    def test_moving_average(self):
        for data in self.data:
            f, n_components = get_decomposition('moving_average')
            trend, seasonal = f(data)
            self.assertEqual(trend.shape, data.shape)
            self.assertEqual(seasonal.shape, data.shape)
            self.assertEqual(n_components, 2)
            self.assertTrue(np.allclose(trend + seasonal, data))

    def test_differential_decomposition(self):
        for data in self.data:
            f, n_components = get_decomposition('differential')
            trend, seasonal = f(data)
            self.assertEqual(trend.shape, data.shape)
            self.assertEqual(seasonal.shape, data.shape)
            self.assertEqual(n_components, 2)
            self.assertTrue(np.allclose(trend + seasonal, data))

    def test_classic_decomposition(self):
        for data in self.data:
            f, n_components = get_decomposition('classic')
            trend, seasonal, resid = f(data)
            self.assertEqual(trend.shape, data.shape)
            self.assertEqual(seasonal.shape, data.shape)
            self.assertEqual(resid.shape, data.shape)
            self.assertEqual(n_components, 3)
            self.assertTrue(np.allclose(trend + seasonal + resid, data))

    def test_unknown_decomposition(self):
        with self.assertRaises(ValueError):
            get_decomposition('unknown')
