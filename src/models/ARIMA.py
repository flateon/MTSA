import warnings
from multiprocessing import Pool

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import trange, tqdm

from src.models.base import MLForecastModel
from statsmodels.tsa.arima.model import ARIMA as ARIMABase
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic


class ARIMA(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.models = []
        self.args = args
        self.order = args.order if hasattr(args, 'order') else None
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

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
            x = x[-1000:]
            res = arma_order_select_ic(x, ic='bic')
            p, q = res.bic_min_order
            order.append((p, d, q))
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

    def _forecast_channel(self, X):
        pred = np.zeros((self.pred_len, X.shape[-1]))
        for channel in range(X.shape[-1]):
            x = X[:, channel]
            pred[:, channel] = self.models[channel].apply(x).forecast(self.pred_len)
        return pred

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        self.pred_len = pred_len
        with Pool() as pool:
            pred = np.array(list(tqdm(pool.imap(self._forecast_channel, X, chunksize=64), total=len(X), leave=False)))
        return pred

    # def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
    #     pred = np.zeros((X.shape[0], pred_len, X.shape[-1]))
    #     for channel in trange(X.shape[-1], leave=False):
    #         for sample in trange(X.shape[0], leave=False):
    #             x = X[sample, :, channel]
    #             pred[sample, :, channel] = self.models[channel].apply(x).forecast(pred_len)
    #     return pred
