import numpy as np

from src.models.base import MLForecastModel
from src.models.baselines import LinearRegression

from src.utils.decomposition import moving_average


class DLinear(MLForecastModel):
    def __init__(self, args, individual=True) -> None:
        super().__init__()
        self.model_trend = LinearRegression()
        self.model_seasonal = LinearRegression()
        self.individual = individual

    def _fit(self, X: np.ndarray) -> None:
        trend, seasonal = moving_average(X)
        if self.individual:
            trend = trend.transpose((0, 2, 1)).reshape(-1, trend.shape[1])[..., np.newaxis]
            seasonal = seasonal.transpose((0, 2, 1)).reshape(-1, seasonal.shape[1])[..., np.newaxis]

        self.model_trend.fit(trend)
        self.model_seasonal.fit(seasonal)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        trend, seasonal = moving_average(X)
        if self.individual:
            trend = trend.transpose((0, 2, 1)).reshape(-1, trend.shape[1])[..., np.newaxis]
            seasonal = seasonal.transpose((0, 2, 1)).reshape(-1, seasonal.shape[1])[..., np.newaxis]

        pred = self.model_trend.forecast(trend, pred_len) + self.model_seasonal.forecast(seasonal, pred_len)
        if self.individual:
            pred = pred.reshape(X.shape[0], X.shape[2], pred_len).transpose((0, 2, 1))
        return pred
