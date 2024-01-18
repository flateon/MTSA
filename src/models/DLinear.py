from pathlib import Path
from typing import Any

import numpy as np
from lightning.pytorch.callbacks import EarlyStopping
from numpy.lib.stride_tricks import sliding_window_view
from lightning.pytorch.loggers import WandbLogger

from src.models.DLbase import DLDataset
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
from torch.utils.data import DataLoader
import logging

# configure logging at the root level of Lightning
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)


class DLinear(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.individual = args.individual
        self.period = args.period
        self.args = args
        self.decomposition, self.n_components = get_decomposition(args.decomposition)
        self.model = DLinearModel(self.args, self.args.seq_len, self.args.pred_len, 1, self.n_components,
                                  lambda x: self.decomposition(x, self.period))

        logger = WandbLogger(project='MTSA') if self.args.log else False
        self.trainer = L.Trainer(max_steps=100000, max_epochs=self.args.epochs, enable_progress_bar=False,
                                 logger=logger, enable_checkpointing=False,
                                 callbacks=[EarlyStopping(monitor='val_loss')])

    def _fit(self, X: np.ndarray | list[np.ndarray], val_X=None) -> None:
        global n_channels
        if isinstance(X, np.ndarray):
            X = [X]
        for i, x in enumerate(X):
            n_samples, train_len, n_channels = x.shape
            if not self.individual:
                X[i] = x.transpose((0, 2, 1)).reshape(-1, train_len, 1)
                n_channels = 1

        if isinstance(val_X, np.ndarray):
            val_X = [val_X]
        for i, x in enumerate(val_X):
            n_samples, train_len, n_channels = x.shape
            if not self.individual:
                val_X[i] = x.transpose((0, 2, 1)).reshape(-1, train_len, 1)
                n_channels = 1

        seq_len = self.args.seq_len
        pred_len = self.args.pred_len

        if self.individual:
            self.model = DLinearModel(self.args, seq_len, pred_len, n_channels, self.n_components,
                                      lambda x: self.decomposition(x, self.period))

        train_loader = DataLoader(DLDataset(X, seq_len, pred_len, 'train'),
                                  batch_size=64, shuffle=True, pin_memory=True)
        val_loader = DataLoader(DLDataset(val_X, seq_len, pred_len, 'train'),
                                batch_size=1024, shuffle=False, pin_memory=True)
        self.trainer.fit(self.model, train_loader, val_loader)
        import wandb
        wandb.finish()

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        n_samples, seq_len, n_channels = X.shape
        if not self.individual:
            X = X.transpose((0, 2, 1)).reshape(-1, seq_len, 1)

        test_loader = DataLoader(DLDataset(X, seq_len, pred_len, 'predict'),
                                 batch_size=1024, shuffle=False, pin_memory=True)

        predictions = self.trainer.predict(self.model, test_loader)
        pred = torch.cat(predictions, dim=0).numpy()

        if not self.individual:
            pred = pred.reshape(n_samples, n_channels, pred_len).transpose((0, 2, 1))
        return pred

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


class DLinearModel(L.LightningModule):
    def __init__(self, args, seq_len, pred_len, n_channel, n_components, decomposition):
        super().__init__()
        self.args = args
        self.n_channel = n_channel
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomposition = decomposition
        self.models = nn.ModuleList(
            nn.ModuleList(nn.Linear(seq_len, pred_len) for _ in range(n_channel)) for _ in range(n_components))

    def forward(self, x):
        """
        :param x: tensor has the shape=(batch_size, seq_len, channels)
        :return: shape=(batch_size, pred_len, channels)
        """
        batch_size, _, _ = x.shape
        x_decomposed = self.decomposition(x)

        # mean = [torch.mean(x_, dim=1, keepdim=True) for x_ in x_decomposed]
        # x_decomposed = [x - m for x, m in zip(x_decomposed, mean)]

        output = torch.empty((batch_size, self.pred_len, self.n_channel), device=x_decomposed[0].device)
        for i in range(self.n_channel):
            output[..., i] = sum((model[i](x[..., i]) for model, x in zip(self.models, x_decomposed)))
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = nn.functional.mse_loss(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = nn.functional.mse_loss(pred, y)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0) -> Any:
        return self.forward(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]
