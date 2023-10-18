import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel


class ZeroForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        return np.zeros((X.shape[0], pred_len))


class MeanForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        mean = np.mean(X, axis=-1).reshape(X.shape[0], 1)
        return np.repeat(mean, pred_len, axis=1)


class LinearRegression(MLForecastModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.fitted = False
        # X: np.ndarray, shape=(n_samples, timestamps)
        self.X = None

    def _fit(self, X: np.ndarray):
        """
        :param:
        X: np.ndarray, shape=(n_samples, timestamps, channels)
        """
        # self.X = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
        self.X = X[..., -1]

    def _forecast(self, X_test: np.ndarray, pred_len) -> np.ndarray:
        """
        :param:
        X_test: shape=(n_samples, timestamps)
        pred_len: int
        :return: forecast: shape=(n_samples, pred_len)
        """
        train_len = X_test.shape[-1]
        window_len = train_len + pred_len

        # shape=(n, window_len)
        train_data = np.concatenate([sliding_window_view(x, window_len) for x in self.X])
        x, y = np.split(train_data, [96], axis=1)

        x = np.c_[np.ones(len(x)), x]

        # shape=(train_len + 1, pred_len)
        weight = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)

        X_test = np.c_[np.ones(len(X_test)), X_test]

        return X_test.dot(weight)


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
        X_test: shape=(n_samples, timestamps)
        pred_len: int
        :return: forecast: shape=(n_samples, pred_len)
        """
        self.target = X[:, 0]
        for t in X.T[1:]:
            # self.target = self.target * self.ew + t * (1 - self.ew)
            self.target = t + self.ew * (self.target - t)

        return np.repeat(self.target[:, np.newaxis], pred_len, axis=1)
