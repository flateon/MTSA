import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel
from src.models.baselines import LinearRegression

from src.utils.decomposition import moving_average


class DLinearClosedForm(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.model_trend = LinearRegression()
        self.model_seasonal = LinearRegression()
        self.individual = args.individual

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


import lightning as L
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import logging

# configure logging at the root level of Lightning
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)


class DLinear(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = None
        self.individual = args.individual

    def _fit(self, X: np.ndarray, args) -> None:
        n_samples, train_len, n_channels = X.shape
        if not self.individual:
            X = X.transpose((0, 2, 1)).reshape(-1, train_len, 1)
            n_channels = 1

        seq_len = args.seq_len
        pred_len = args.pred_len
        window_len = seq_len + pred_len

        train_data = np.concatenate([sliding_window_view(x, (window_len, n_channels)) for x in X])[:, 0, ...]
        x_windowed, y_windowed = np.split(train_data, [seq_len], axis=1)
        x_trend, x_seasonal = moving_average(x_windowed)

        x_trend = torch.tensor(x_trend, dtype=torch.float32)
        x_seasonal = torch.tensor(x_seasonal, dtype=torch.float32)
        y = torch.tensor(y_windowed, dtype=torch.float32)

        self.model = DLinearModel(seq_len, pred_len, n_channels)
        train_loader = DataLoader(TensorDataset(x_trend, x_seasonal, y), batch_size=32, shuffle=True)
        trainer = L.Trainer(max_epochs=5, accelerator='cpu', enable_progress_bar=False)
        trainer.fit(self.model, train_loader)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        n_samples, seq_len, n_channels = X.shape
        if not self.individual:
            X = X.transpose((0, 2, 1)).reshape(-1, seq_len, 1)

        x_trend, x_seasonal = moving_average(X)
        x_trend = torch.tensor(x_trend, dtype=torch.float32)
        x_seasonal = torch.tensor(x_seasonal, dtype=torch.float32)
        with torch.no_grad():
            pred = self.model(x_trend, x_seasonal).numpy()

        if not self.individual:
            pred = pred.reshape(n_samples, n_channels, pred_len).transpose((0, 2, 1))
        return pred


class DLinearModel(L.LightningModule):
    def __init__(self, seq_len, pred_len, n_channel):
        super().__init__()
        self.n_channel = n_channel
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.trend = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(n_channel)])
        self.seasonal = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(n_channel)])

    def forward(self, x_trend, x_seasonal):
        """
        :param x_trend: shape=(batch_size, seq_len, channels)
        :param x_seasonal: shape=(batch_size, seq_len, channels)
        :return: shape=(batch_size, pred_len, channels)
        """
        batch_size, _, _ = x_trend.shape

        output = torch.empty((batch_size, self.pred_len, self.n_channel))
        for i in range(self.n_channel):
            output[..., i] = self.trend[i](x_trend[..., i]) + self.seasonal[i](x_seasonal[..., i])
        return output

    def training_step(self, batch, batch_idx):
        x_t, x_s, y = batch
        pred = self.forward(x_t, x_s)
        loss = nn.functional.mse_loss(pred, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]
