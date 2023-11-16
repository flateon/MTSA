import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel
from src.models.baselines import LinearRegression

from src.utils.decomposition import moving_average


class DLinear(MLForecastModel):
    def __init__(self, individual=True) -> None:
        super().__init__()
        self.model_trend = LinearRegression()
        self.model_seasonal = LinearRegression()
        self.individual = individual

    def _fit(self, X: np.ndarray, args) -> None:
        # trend, seasonal = moving_average(X)
        # if not self.individual:
        #     trend = trend.transpose((0, 2, 1)).reshape(-1, trend.shape[1], 1)
        #     seasonal = seasonal.transpose((0, 2, 1)).reshape(-1, seasonal.shape[1], 1)
        #
        # self.model_trend.fit(trend, arg)
        # self.model_seasonal.fit(seasonal, arg)
        n_samples, train_len, n_channels = X.shape
        if not self.individual:
            X = X.transpose((0, 2, 1)).reshape(-1, train_len, 1)
            n_channels = 1

        seq_len = args.seq_len
        pred_len = args.pred_len
        window_len = seq_len + pred_len

        train_data = np.concatenate([sliding_window_view(x, (window_len, n_channels)) for x in X])[:, 0, ...]
        x_windowed, y_windowed = np.split(train_data, [seq_len], axis=1)

        x_t, x_s = moving_average(x_windowed)
        y_t, y_s = moving_average(y_windowed)

        self.model_trend.calc_weight(x_t, y_t)
        self.model_seasonal.calc_weight(x_s, y_s)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        n_samples, seq_len, n_channels = X.shape
        if not self.individual:
            X = X.transpose((0, 2, 1)).reshape(-1, seq_len, 1)

        trend, seasonal = moving_average(X)

        pred = self.model_trend.forecast(trend, pred_len) + self.model_seasonal.forecast(seasonal, pred_len)

        if not self.individual:
            pred = pred.reshape(n_samples, n_channels, pred_len).transpose((0, 2, 1))
        return pred
