import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel
from src.models.baselines import LinearRegression


class FLinear(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.model_f = [LinearRegression(args), LinearRegression(args)]
        self.model_t = LinearRegression(args)
        if args.fl_weight == 'time':
            self.weight = np.zeros((720, 1))
        elif args.fl_weight == 'freq':
            self.weight = np.ones((720, 1))
        elif args.fl_weight == 'linear':
            self.weight = np.linspace(0, 1, 720)[:, np.newaxis]
        elif args.fl_weight == 'constant':
            self.weight = np.ones((720, 1)) * 0.5

    def _fit(self, X: np.ndarray, val_X=None) -> None:
        n_samples, train_len, n_channels = X.shape

        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
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


import lightning as L
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import logging

# configure logging at the root level of Lightning
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)


class FLinearGD(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = None
        self.args = args
        if not hasattr(args, 'fl_weight'):
            self.weight = 'learn'
        else:
            assert args.fl_weight in ('time', 'freq', 'linear', 'constant', 'learn')
            self.weight = args.fl_weight

        self.individual = args.individual

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

        x = torch.tensor(x_windowed, dtype=torch.float32)
        y = torch.tensor(y_windowed, dtype=torch.float32)

        self.model = FLinearModel(seq_len, pred_len, self.weight)
        # self.model = torch.compile(self.model)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True, num_workers=0)
        trainer = L.Trainer(max_steps=50000, max_epochs=10, enable_progress_bar=True, logger=False,
                            enable_checkpointing=False)
        trainer.fit(self.model, train_loader)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        n_samples, seq_len, n_channels = X.shape
        if not self.individual:
            X = X.transpose((0, 2, 1)).reshape(-1, seq_len, 1)

        with torch.inference_mode():
            pred = self.model(torch.tensor(X, dtype=torch.float32)).numpy()

        if not self.individual:
            pred = pred.reshape(n_samples, n_channels, pred_len).transpose((0, 2, 1))
        return pred


class FLinearModel(L.LightningModule):
    def __init__(self, seq_len, pred_len, weight):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model_t = nn.Linear(seq_len, pred_len)
        self.model_f = nn.Linear(seq_len // 2 + 1, pred_len // 2 + 1, dtype=torch.cfloat)
        if weight == 'time':
            self.weight = nn.Parameter(torch.zeros((pred_len,), requires_grad=False))
        elif weight == 'freq':
            self.weight = nn.Parameter(torch.ones((pred_len,), requires_grad=False))
        elif weight == 'linear':
            self.weight = nn.Parameter(torch.linspace(0, 1, 720)[:pred_len], requires_grad=False)
        elif weight == 'constant':
            self.weight = nn.Parameter(torch.ones((pred_len,) * 0.5, requires_grad=False))
        elif weight == 'learn':
            self.weight = nn.Parameter(torch.linspace(0, 1, 720)[:pred_len], requires_grad=True)

    def forward(self, X):
        """
        :param X: tensor shape=(batch_size, seq_len, channels)
        :return: shape=(batch_size, pred_len, channels)
        """
        batch_size, seq_len, n_channel = X.shape
        # mean = X.mean(dim=1, keepdim=True)
        # X = X - mean

        output = torch.empty((batch_size, self.pred_len, n_channel), device=X.device)
        X_fft = torch.fft.rfftn(X, dim=(1,))
        X_fft[:, 0, :] = 0
        for i in range(n_channel):
            pred_t = self.model_t(X[..., i])
            pred_f = torch.fft.irfftn(self.model_f(X_fft[..., i]), dim=(1,))

            output[..., i] = pred_f * self.weight + pred_t * (1 - self.weight)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = nn.functional.mse_loss(pred, y)
        # self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]
