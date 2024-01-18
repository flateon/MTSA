import time
import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.models.base import MLForecastModel
from src.utils.tools import EarlyStopping, adjust_learning_rate


class DLDataset(Dataset):
    def __init__(self, X: Union[list[np.ndarray | torch.Tensor], np.ndarray, torch.Tensor], seq_len: int, pred_len: int,
                 mode: str = 'train'):
        """
        :param X: list of np.ndarray or torch.Tensor
        :param seq_len: int
        :param pred_len: int
        :param mode: str
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode
        if self.mode == 'train':
            if isinstance(X, list):
                self.data = X
            elif isinstance(X, np.ndarray) or isinstance(X, torch.Tensor):
                self.data = [x for x in X]
            # self.data: tuple of 2d np.ndarray or torch.Tensor
            self.data = sum([tuple(x_2d for x_2d in x) if len(x.shape) == 3 else (x,) for x in self.data], ())
            self.size = [len(x) - self.seq_len - self.pred_len + 1 for x in self.data]
            cumsum = np.cumsum(self.size)

            def idx2list_idx(idx):
                list_idx = np.searchsorted(cumsum, idx, side='right')
                bias = 0 if list_idx == 0 else cumsum[list_idx - 1]
                return list_idx, idx - bias

            self.idx2list_idx = idx2list_idx
        else:
            self.data = X

    def __len__(self):
        if self.mode == 'predict':
            return len(self.data)
        else:
            return sum(self.size)

    def __getitem__(self, idx):
        if self.mode == 'predict':
            return self.data[idx]
        else:
            list_idx, idx = self.idx2list_idx(idx)
            data = self.data[list_idx]
            x = data[idx: idx + self.seq_len]
            y = data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
            return x, y


class DLForecastModel(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.optimizer = None
        self.model = None
        self.args = args
        if self.args.device == 'cpu':
            self.device = 'cpu'
        else:
            self.device = f'cuda:{self.args.device}'
        self.criterion = nn.MSELoss()

    def _fit(self, train_X: np.ndarray, val_X=None):
        train_loader = DataLoader(DLDataset(train_X, self.args.seq_len, self.args.pred_len),
                                  batch_size=self.args.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(DLDataset(val_X, self.args.seq_len, self.args.pred_len),
                                batch_size=self.args.batch_size,
                                shuffle=False)

        path = os.path.join('checkpoints', f'{self.args.model}_{self.args.dataset}_{self.args.pred_len}')
        if not os.path.exists(path):
            os.makedirs(path)

        # train
        self.model.train()
        train_epochs = self.args.epochs
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        for epoch in range(train_epochs):
            train_loss = 0
            epoch_time = time.time()
            for batch_idx, (batch_x, batch_y) in enumerate(tqdm(train_loader, leave=False)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device.split(':')[0], enabled=self.args.use_amp):
                    outputs = self.model(batch_x)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :]
                    loss = self.criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                train_loss += loss.item()
            train_loss = train_loss / len(train_loader)

            # validate
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    outputs = self.model(batch_x)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :]
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            print("Epoch: {} cost time: {} Train Loss: {} Val Loss: {}".format(epoch + 1, time.time() - epoch_time,
                                                                               train_loss, val_loss))
            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.args)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        test_loader = DataLoader(DLDataset(X, self.args.seq_len, self.args.pred_len, mode='predict'),
                                 batch_size=self.args.batch_size,
                                 shuffle=False)
        # predict
        self.model.eval()
        fore = []
        with torch.no_grad():
            for batch_idx, (batch_x) in enumerate(tqdm(test_loader, leave=False)):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:, :]
                fore.append(outputs.cpu().numpy())
        fore = np.concatenate(fore, axis=0)
        return fore

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
