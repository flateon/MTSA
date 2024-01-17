import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel
from src.models.baselines import LinearRegression

from src.utils.decomposition import get_decomposition


class DLinearClosedForm(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.decomposition, self.n_components = get_decomposition(args.decomposition)
        self.models = [LinearRegression(args) for _ in range(self.n_components)]
        self.individual = args.individual
        self.period = args.period
        self.args = args

    def _fit(self, X: np.ndarray, val_X=None) -> None:
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

        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        window_len = seq_len + pred_len

        train_data = np.concatenate([sliding_window_view(x, (window_len, n_channels)) for x in X])[:, 0, ...]
        x_windowed, y_windowed = np.split(train_data, [seq_len], axis=1)

        x_decomposed = self.decomposition(x_windowed, self.period)
        y_decomposed = self.decomposition(y_windowed, self.period)
        for x, y, model in zip(x_decomposed, y_decomposed, self.models):
            model.fit_windowed(x, y)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        n_samples, seq_len, n_channels = X.shape
        if not self.individual:
            X = X.transpose((0, 2, 1)).reshape(-1, seq_len, 1)

        decomposed = self.decomposition(X, self.period)

        pred = sum((model.forecast(x, pred_len) for model, x in zip(self.models, decomposed)))

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
        self.period = args.period
        self.args = args
        self.decomposition, self.n_components = get_decomposition(args.decomposition)

    def _fit(self, X: np.ndarray, val_X=None) -> None:
        n_samples, train_len, n_channels = X.shape
        if not self.individual:
            X = X.transpose((0, 2, 1)).reshape(-1, train_len, 1)
            n_channels = 1

        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        window_len = seq_len + pred_len

        train_data = np.concatenate([sliding_window_view(x, (window_len, n_channels)) for x in X])[:, 0, ...]
        x_windowed, y_windowed = np.split(train_data, [seq_len], axis=1)

        x_decomposed = self.decomposition(x_windowed, period=self.period)
        x_decomposed = (torch.tensor(x, dtype=torch.float32) for x in x_decomposed)
        y = torch.tensor(y_windowed, dtype=torch.float32)

        self.model = DLinearModel(seq_len, pred_len, n_channels, self.n_components)
        train_loader = DataLoader(TensorDataset(*x_decomposed, y), batch_size=32, shuffle=True)
        trainer = L.Trainer(max_steps=50000, max_epochs=10, enable_progress_bar=True, logger=False, enable_checkpointing=False)
        trainer.fit(self.model, train_loader)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        n_samples, seq_len, n_channels = X.shape
        if not self.individual:
            X = X.transpose((0, 2, 1)).reshape(-1, seq_len, 1)

        x_decomposed = self.decomposition(X, period=self.period)
        with torch.inference_mode():
            x_decomposed = tuple(torch.tensor(x, dtype=torch.float32) for x in x_decomposed)
            pred = self.model(x_decomposed).numpy()

        if not self.individual:
            pred = pred.reshape(n_samples, n_channels, pred_len).transpose((0, 2, 1))
        return pred


class DLinearModel(L.LightningModule):
    def __init__(self, seq_len, pred_len, n_channel, n_components):
        super().__init__()
        self.n_channel = n_channel
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.models = nn.ModuleList(
            nn.ModuleList(nn.Linear(seq_len, pred_len) for _ in range(n_channel)) for _ in range(n_components))

    def forward(self, x_decomposed):
        """
        :param x_decomposed: list of tensor,each tensor has the same shape=(batch_size, seq_len, channels)
        :return: shape=(batch_size, pred_len, channels)
        """
        batch_size, _, _ = x_decomposed[0].shape

        output = torch.empty((batch_size, self.pred_len, self.n_channel), device=x_decomposed[0].device)
        for i in range(self.n_channel):
            output[..., i] = sum((model[i](x[..., i]) for model, x in zip(self.models, x_decomposed)))
        return output

    def training_step(self, batch, batch_idx):
        x_decomposed = batch[:-1]
        y = batch[-1]
        pred = self.forward(x_decomposed)
        loss = nn.functional.mse_loss(pred, y)
        # self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]
