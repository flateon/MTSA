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
        target = np.array([1, 3, 5, 7])
        self.assertAlmostEqual(mape(predict, target), 41.9047619047619)

    def test_smape(self):
        predict = np.array([2, 4, 6, 8])
        target = np.array([1, 3, 5, 7])
        self.assertAlmostEqual(smape(predict, target), 31.68831168831169)

    def test_mase(self):
        predict = np.array([2, 4, 6, 8])
        target = np.array([1, 3, 5, 7])
        m = 2  # You need to specify the seasonality parameter 'm'
        self.assertAlmostEqual(mase(predict, target, m), 0.25)

    def test_metrics(self):
        predict = np.array([2, 4, 6, 8])
        target = np.array([1, 3, 5, 7])
        m = 2  # You need to specify the seasonality parameter 'm'
        mse_, mae_, mape_, smape_, mase_ = metrics(predict, target, m)
        self.assertAlmostEqual(mse_, 1.0)
        self.assertAlmostEqual(mae_, 1.0)
        self.assertAlmostEqual(mape_, 41.9047619047619)
        self.assertAlmostEqual(smape_, 31.68831168831169)
        self.assertAlmostEqual(mase_, 0.25)
