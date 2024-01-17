from multiprocessing import Pool

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import trange, tqdm

from src.models.base import MLForecastModel
from statsmodels.tsa.forecasting.theta import ThetaModel


class ThetaMethod(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.period = args.period
        import warnings
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def _fit(self, X: np.ndarray, val_X=None) -> None:
        pass

    def _forecast_channel(self, X):
        pred = np.zeros((self.pred_len, X.shape[-1]))
        for channel in range(X.shape[-1]):
            x = X[:, channel]
            try:
                pred[:, channel] = ThetaModel(x, period=self.period, use_test=True).fit().forecast(self.pred_len)
            except ValueError:
                pred[:, channel] = ThetaModel(x, period=self.period, deseasonalize=False).fit().forecast(self.pred_len)
        return pred

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        # pred = np.zeros((X.shape[0], pred_len, X.shape[-1]))
        # for sample in trange(X.shape[0], leave=False):
        #     pred[sample] = self._forecast_channel(X[sample])
        self.pred_len = pred_len
        with Pool() as pool:
            pred = np.array(list(tqdm(pool.imap(self._forecast_channel, X, chunksize=8), total=len(X), leave=False)))
        return pred
