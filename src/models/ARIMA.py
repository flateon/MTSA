import numpy as np
from tqdm import trange

from src.models.base import MLForecastModel
from statsmodels.tsa.arima.model import ARIMA as ARIMABase
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic


class ARIMA(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.models = []
        self.args = args
        self.order = args.order if hasattr(args, 'order') else None

    def get_order(self, X):
        order = []
        for channel in range(X.shape[-1]):
            x = X[..., channel]
            d = 0
            while True:
                adfstat, pvalue, usedlag, nobs, critvalues, icbest = adfuller(x)
                if pvalue < 0.05:
                    break
                else:
                    d += 1
                    x = np.diff(x)

            res = arma_order_select_ic(x, ic='bic')
            p, q = res.bic_min_order
            order.append((p, d, q))
        print(order)
        return order

    def _fit(self, X: np.ndarray, args) -> None:
        if len(X.shape) == 3:
            X = X[0, ...]
        if self.order is None:
            self.order = self.get_order(X)
        else:
            self.order = [self.order] * X.shape[-1]

        for channel in trange(X.shape[-1], leave=False, desc='Fitting'):
            x = X[:, channel]
            self.models.append(ARIMABase(x, order=self.order[channel]).fit())

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        pred = np.zeros((X.shape[0], pred_len, X.shape[-1]))
        for channel in trange(X.shape[-1], leave=False):
            for sample in trange(X.shape[0], leave=False):
                x = X[sample, :, channel]
                pred[sample, :, channel] = self.models[channel].apply(x).forecast(pred_len)
        return pred
