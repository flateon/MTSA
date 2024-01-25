import numpy as np
import unittest

from src.models.DLinear import DLinear, DLinearClosedForm
from src.models.ResidualModel import FLinear
from src.models.TsfKNN import TsfKNN
from src.models.baselines import ZeroForecast, MeanForecast, LinearRegression, ExponentialSmoothing
from src.models.base import MLForecastModel
from src.models.ARIMA import ARIMA
from src.models.ThetaMethod import ThetaMethod
from argparse import Namespace as Args


class TestModels(unittest.TestCase):
    def setUp(self):
        self.seq_len = 96
        self.pred_len = 32
        self.n_channels = 3
        self.n_train = 500
        self.n_test = 3
        # (n_samples, timestamps, channels)
        self.X = np.random.rand(1, self.n_train, self.n_channels) + np.arange(0, self.n_train)[:, np.newaxis].repeat(
            self.n_channels, axis=-1)
        self.X = self.X.astype(np.float32)
        # (n_samples, timestamps, channels)
        self.X_test = np.random.rand(self.n_test, self.seq_len, self.n_channels).astype(np.float32)
        self.fore_shape = (len(self.X_test), self.pred_len, self.n_channels)
        self.args = Args(seq_len=self.seq_len, pred_len=self.pred_len, individual=False, period=24, log=False, epochs=1,
                         lr=1e-4)

    def test_flinear(self):
        self.args.fl_weight = 'constant'
        model = FLinear(self.args)
        model.fit(self.X, self.args)
        forecast = model.forecast(self.X_test, self.pred_len)

        self.assertEqual(forecast.shape, self.fore_shape)

    def test_arima(self):
        model = ARIMA(self.args)
        model.fit(self.X, self.args)
        forecast = model.forecast(self.X_test, self.pred_len)

        self.assertEqual(forecast.shape, self.fore_shape)

    def test_theta(self):
        model = ThetaMethod(self.args)
        model.fit(self.X, self.args)
        forecast = model.forecast(self.X_test, self.pred_len)

        self.assertEqual(forecast.shape, self.fore_shape)

    def test_zero_forecast(self):
        model = ZeroForecast()
        model.fit(self.X, self.args)
        forecast = model.forecast(self.X_test, self.pred_len)

        self.assertEqual(forecast.shape, self.fore_shape)
        self.assertTrue(np.all(forecast == 0))

    def test_mean_forecast(self):
        model = MeanForecast()
        model.fit(self.X, self.args)
        forecast = model.forecast(self.X_test, self.pred_len)

        self.assertEqual(forecast.shape, self.fore_shape)
        mean = np.mean(self.X_test, axis=1, keepdims=True)
        expected_forecast = np.repeat(mean, self.pred_len, axis=1)
        self.assertTrue(np.all(forecast == expected_forecast))

    def test_linear_regression(self):
        for individual in (True, False):
            setattr(self.args, 'individual', individual)
            model = LinearRegression(self.args)
            model.fit(self.X, self.args)
            forecast = model.forecast(self.X_test, self.pred_len)

            self.assertEqual(forecast.shape, self.fore_shape)

    def test_d_linear(self):
        for individual in (True, False):
            for decomposition in ('moving_average', 'classic'):
                setattr(self.args, 'individual', individual)
                setattr(self.args, 'decomposition', decomposition)
                model = DLinear(self.args)
                model.fit(self.X, self.X)
                forecast = model.forecast(self.X_test, self.pred_len)

                self.assertEqual(forecast.shape, self.fore_shape)

    def test_d_linear_closed_form(self):
        for individual in (True, False):
            for decomposition in ('moving_average', 'differential', 'classic'):
                setattr(self.args, 'individual', individual)
                setattr(self.args, 'decomposition', decomposition)
                model = DLinearClosedForm(self.args)
                model.fit(self.X, self.args)
                forecast = model.forecast(self.X_test, self.pred_len)

                self.assertEqual(forecast.shape, self.fore_shape)

    def test_exponential_smoothing(self):
        model = ExponentialSmoothing(Args(ew=0.5))
        model.fit(self.X, self.args)
        forecast = model.forecast(self.X_test, self.pred_len)

        self.assertEqual(forecast.shape, self.fore_shape)
        # TODO add value assert

    def test_ml_forecast(self):
        model = MLForecastModel()
        self.assertRaises(ValueError, model.forecast, None, None)
        self.assertRaises(NotImplementedError, model.fit, None, self.args)
        model.fitted = True
        self.assertRaises(NotImplementedError, model.forecast, None, None)

    def test_tsf_knn(self):
        for m in ('MIMO', 'recursive'):
            for e in ('lag', 'fourier'):
                for k in ('brute_force', 'lsh'):
                    for d in ('euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine', 'decompose', 'zero'):
                        for decomposition in ('moving_average', 'differential', 'classic'):
                            args = Args(n_neighbors=3, distance=d, msas=m, knn=k, num_bits=4, num_hashes=2,
                                        embedding=e, decomposition=decomposition, tau=1, seq_len=self.seq_len,
                                        pred_len=self.pred_len)
                            model = TsfKNN(args)
                            model.fit(self.X, args)
                            forecast = model.forecast(self.X_test, self.pred_len)

                            self.assertEqual(forecast.shape, self.fore_shape)
                            # TODO add value assert
                            if d != 'decompose':
                                break

        # lsh test if the number of candidates is less than k
        args = Args(n_neighbors=3, distance='euclidean', msas='MIMO', knn='lsh', num_bits=12, num_hashes=2,
                    embedding='lag', tau=1, seq_len=self.seq_len, pred_len=self.pred_len)
        model = TsfKNN(args)
        model.fit(self.X, args)
        forecast = model.forecast(self.X_test, self.pred_len)

        self.assertEqual(forecast.shape, self.fore_shape)

        # test lsh
        _, acc = model.knn.query(self.X_test[0], return_acc=True)
        self.assertTrue(0 <= acc <= 1)

        # test raise
        self.assertRaises(ValueError, TsfKNN,
                          Args(n_neighbors=3, distance='zero', msas='MIMO', knn='brute_force', embedding='foo', tau=1,
                               seq_len=self.seq_len, pred_len=self.pred_len))
        self.assertRaises(ValueError, TsfKNN,
                          Args(n_neighbors=3, distance='zero', msas='MIMO', knn='foo', embedding='lag', tau=1,
                               seq_len=self.seq_len, pred_len=self.pred_len))
        self.assertRaises(ValueError, TsfKNN,
                          Args(n_neighbors=3, distance='zero', msas='foo', knn='brute_force', embedding='lag', tau=1,
                               seq_len=self.seq_len, pred_len=self.pred_len))
        self.assertRaises(ValueError, TsfKNN,
                          Args(n_neighbors=3, distance='foo', msas='MIMO', knn='brute_force', embedding='lag', tau=1,
                               seq_len=self.seq_len, pred_len=self.pred_len))
