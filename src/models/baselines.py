import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel


class ZeroForecast(MLForecastModel):
    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        return np.zeros((X.shape[0], pred_len, X.shape[2]))


class MeanForecast(MLForecastModel):
    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        mean = np.mean(X, axis=1, keepdims=True)
        return np.repeat(mean, pred_len, axis=1)


class LinearRegression(MLForecastModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # X: np.ndarray, shape=(n_samples, timestamps, n_channels)
        self.X = None

    def _fit(self, X: np.ndarray):
        """
        :param:
        X: np.ndarray, shape=(n_samples, timestamps, n_channels)
        """
        # self.X = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
        self.X = X.reshape(X.shape[0], -1)

    def _forecast(self, X_test: np.ndarray, pred_len) -> np.ndarray:
        """
        :param:
        X_test: shape=(n_samples, timestamps, channels)
        pred_len: int
        :return: forecast: shape=(n_samples, pred_len, channels)
        """
        n_samples, train_len, n_channels = X_test.shape
        X_test = X_test.reshape(n_samples, -1)

        window_len = (train_len + pred_len) * n_channels

        # shape=(n, window_len)
        train_data = np.concatenate([sliding_window_view(x, window_len) for x in self.X])
        x, y = np.split(train_data, [train_len * n_channels], axis=1)

        x = np.c_[np.ones(len(x)), x]

        # shape=(train_len + 1, pred_len)
        weight = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)

        X_test = np.c_[np.ones(len(X_test)), X_test]

        return X_test.dot(weight).reshape(-1, pred_len, n_channels)


class ExponentialSmoothing(MLForecastModel):
    def __init__(self, arg, *args, **kwargs) -> None:
        super().__init__()
        self.fitted = False
        self.ew = arg.ew
        self.target = None

    def _fit(self, X: np.ndarray):
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
