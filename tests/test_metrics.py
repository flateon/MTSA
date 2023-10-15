import unittest
import numpy as np
from src.utils.metrics import *


class TestMetrics(unittest.TestCase):
    def test_mse(self):
        predict = np.array([2, 4, 6, 8])
        target = np.array([1, 3, 5, 7])
        self.assertAlmostEqual(mse(predict, target), 1.0)

    def test_mae(self):
        predict = np.array([2, 4, 6, 8])
        target = np.array([1, 3, 5, 7])
        self.assertAlmostEqual(mae(predict, target), 1.0)

    def test_mape(self):
        predict = np.array([2, 4, 6, 8])
        target = np.array([1, 2, 3, 4])
        self.assertAlmostEqual(mape(predict, target), 100.0)

    def test_smape(self):
        predict = np.array([2, 4, 6, 8])
        target = np.array([1, 2, 3, 4])
        self.assertAlmostEqual(smape(predict, target), 200 / 3)

    def test_mase(self):
        predict = np.array([2, 4, 6, 8])
        target = np.array([1, 3, 5, 7])
        m = 2  # You need to specify the seasonality parameter 'm'
        self.assertAlmostEqual(mase(predict, target, m), 0.25)


if __name__ == '__main__':
    unittest.main()
