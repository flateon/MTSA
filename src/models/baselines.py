import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel


class ZeroForecast(MLForecastModel):
    def _fit(self, X: np.ndarray, args) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        return np.zeros((X.shape[0], pred_len, X.shape[2]))


class MeanForecast(MLForecastModel):
    def _fit(self, X: np.ndarray, args) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        mean = np.mean(X, axis=1, keepdims=True)
        return np.repeat(mean, pred_len, axis=1)


class LinearRegression(MLForecastModel):
    def __init__(self, args, **kwargs) -> None:
        super().__init__()
        # X: np.ndarray, shape=(n_samples, timestamps, n_channels)
        self.W = None
        self.individual = args.individual

    def _fit(self, X: np.ndarray, args):
        """
        :param:
        X: np.ndarray, shape=(n_samples, timestamps, n_channels)
        """
        n_samples, train_len, n_channels = X.shape
        if not self.individual:
            X = X.transpose((0, 2, 1)).reshape(-1, train_len, 1)
            n_channels = 1

        seq_len = args.seq_len
        pred_len = args.pred_len
        self.pred_len = pred_len

        window_len = seq_len + pred_len

        train_data = np.concatenate([sliding_window_view(x, (window_len, n_channels)) for x in X])[:, 0, ...]
        x_w, y_w = np.split(train_data, [seq_len], axis=1)

        self.fit_windowed(x_w, y_w)

    def fit_windowed(self, X, Y):
        """
        :param:
        X: np.ndarray, shape=(n_samples, seq_len, n_channels)
        Y: np.ndarray, shape=(n_samples, pred_len, n_channels)
        :return:
        self.W: np.ndarray, shape=(n_channels, seq_len + 1, pred_len)
        """
        _, seq_len, n_channels = X.shape
        _, pred_len, _ = Y.shape
        if not self.individual:
            X = X.transpose((0, 2, 1)).reshape(-1, seq_len, 1)
            Y = Y.transpose((0, 2, 1)).reshape(-1, pred_len, 1)
            n_channels = 1
        self.W = np.zeros((n_channels, seq_len + 1, pred_len))

        for i in range(n_channels):
            x, y = X[..., i], Y[..., i]
            x = np.c_[np.ones(len(x)), x]
            self.W[i] = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
        self.fitted = True
        self.pred_len = pred_len
        return self.W

    def _forecast(self, X_test: np.ndarray, pred_len) -> np.ndarray:
        """
        :param:
        X_test: shape=(n_samples, timestamps, channels)
        pred_len: int
        :return: forecast: shape=(n_samples, pred_len, channels)
        """
        n_samples, timestamps, n_channels = X_test.shape
        if not self.individual:
            original_n_samples, timestamps, original_channels = X_test.shape
            X_test = X_test.transpose((0, 2, 1)).reshape(-1, timestamps, 1)
            n_channels = 1
            n_samples = X_test.shape[0]

        X_test_c = np.empty((n_samples, timestamps + 1, n_channels), dtype=X_test.dtype)
        X_test_c[:, 0, :] = 1
        X_test_c[:, 1:, :] = X_test

        pred = np.einsum('nsc,csp->npc', X_test_c, self.W)

        if not self.individual:
            pred = pred.reshape(original_n_samples, original_channels, self.pred_len).transpose((0, 2, 1))

        return pred

class ExponentialSmoothing(MLForecastModel):
    def __init__(self, arg, *args, **kwargs) -> None:
        super().__init__()
        self.fitted = False
        self.ew = arg.ew
        self.target = None

    def _fit(self, X: np.ndarray, args):
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        """
        :param:
        X_test: shape=(n_samples, timestamps, n_channels)
        pred_len: int
        :return: forecast: shape=(n_samples, pred_len, n_channels)
        """
        self.target = X[:, 0, :]
        for i in range(X.shape[1]):
            # self.target = self.target * self.ew + t * (1 - self.ew)
            self.target = X[:, i, :] + self.ew * (self.target - X[:, i, :])

        return np.repeat(self.target[:, np.newaxis, :], pred_len, axis=1)
