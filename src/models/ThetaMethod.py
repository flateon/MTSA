import numpy as np
from tqdm import trange

from src.models.base import MLForecastModel
from statsmodels.tsa.forecasting.theta import ThetaModel


class ThetaMethod(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.period = 24

    def _fit(self, X: np.ndarray, args) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        pred = np.zeros((X.shape[0], pred_len, X.shape[-1]))
        for channel in trange(X.shape[-1], leave=False):
            for sample in trange(X.shape[0], leave=False):
                x = X[sample, :, channel]
                pred[sample, :, channel] = ThetaModel(x, period=self.period).fit().forecast(pred_len)
        return pred
