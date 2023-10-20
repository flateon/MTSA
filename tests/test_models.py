import numpy as np
import unittest

# Import the forecast models you want to test
from src.models.baselines import ZeroForecast, MeanForecast, LinearRegression, ExponentialSmoothing
from src.models.base import MLForecastModel
from tests.test_dataset import Args


class TestModels(unittest.TestCase):
    def setUp(self):
        # (n_samples, timestamps, channels)
        self.X = np.random.rand(1, 100, 7)
        # (n_samples, timestamps)
        self.X_test = np.random.rand(90, 10)
        self.pred_len = 5
        self.fore_shape = (len(self.X_test), self.pred_len)

    def test_zero_forecast(self):
        model = ZeroForecast(None)
        model.fit(self.X)
        forecast = model.forecast(self.X_test, self.pred_len)

        self.assertEqual(forecast.shape, self.fore_shape)
        self.assertTrue(np.all(forecast == 0))

    def test_mean_forecast(self):
        model = MeanForecast(None)
        model.fit(self.X)
        forecast = model.forecast(self.X_test, self.pred_len)

        self.assertEqual(forecast.shape, self.fore_shape)
        mean = np.mean(self.X_test, axis=1, keepdims=True)
        expected_forecast = np.repeat(mean, self.pred_len, axis=1)
        self.assertTrue(np.all(forecast == expected_forecast))

    def test_linear_regression(self):
        model = LinearRegression()
        model.fit(self.X)
        forecast = model.forecast(self.X_test, self.pred_len)

        self.assertEqual(forecast.shape, self.fore_shape)
        # TODO add value assert

    def test_exponential_smoothing(self):
        model = ExponentialSmoothing(Args(ew=0.5))
        model.fit(self.X)
        forecast = model.forecast(self.X_test, self.pred_len)

        self.assertEqual(forecast.shape, self.fore_shape)
        # TODO add value assert

    def test_ml_forecast(self):
        model = MLForecastModel()
        self.assertRaises(ValueError, model.forecast, None, None)
        self.assertRaises(NotImplementedError, model.fit, None)
        model.fitted = True
        self.assertRaises(NotImplementedError, model.forecast, None, None)
