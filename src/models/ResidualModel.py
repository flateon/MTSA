import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel
from src.models.baselines import LinearRegression


class FLinear(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.model_f = [LinearRegression(args), LinearRegression(args)]
        self.model_t = LinearRegression(args)
        if not hasattr(args, 'fl_weight') or args.fl_weight == -1:
            self.weight = np.linspace(0, 1, 720)[:, np.newaxis]
        elif isinstance(args.fl_weight, np.ndarray):
            self.weight = args.fl_weight
        elif isinstance(args.fl_weight, float) and 1 >= args.fl_weight >= 0:
            self.weight = np.ones((720, 1)) * args.fl_weight

    def _fit(self, X: np.ndarray, args) -> None:
        n_samples, train_len, n_channels = X.shape

        seq_len = args.seq_len
        pred_len = args.pred_len
        window_len = seq_len + pred_len

        train_data = np.concatenate([sliding_window_view(x, (window_len, n_channels)) for x in X])[:, 0, ...]
        x_windowed, y_windowed = np.split(train_data, [seq_len], axis=1)

        self.model_t.fit_windowed(x_windowed, y_windowed)

        x_fft = np.fft.rfftn(x_windowed, axes=(1,))
        y_fft = np.fft.rfftn(y_windowed, axes=(1,))
        model_real, model_imag = self.model_f
        model_real.fit_windowed(x_fft.real, y_fft.real)
        model_imag.fit_windowed(x_fft.imag, y_fft.imag)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        x_fft = np.fft.rfftn(X, axes=(1,))
        pred_t = self.model_t.forecast(X, pred_len)

        pred_r = self.model_f[0].forecast(x_fft.real, pred_len)
        pred_i = self.model_f[1].forecast(x_fft.imag, pred_len)
        pred_f = np.fft.irfftn(pred_r + 1j * pred_i, axes=(1,))

        weight = self.weight[:pred_len]
        pred = weight * pred_f + (1 - weight) * pred_t
        return pred
