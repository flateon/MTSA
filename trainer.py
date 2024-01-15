from copy import copy

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.utils.metrics import metrics


class MLTrainer:
    def __init__(self, model, transform, dataset):
        self.model = model
        self.transform = [copy(transform) for _ in range(len(dataset) if isinstance(dataset, list) else 1)]
        self.dataset = dataset

    def train(self, *args, **kwargs):
        X = []
        if not isinstance(self.dataset, list):
            self.dataset = [self.dataset]
        for d, transform in zip(self.dataset, self.transform):
            train_X = d.train_data
            t_X = transform.transform(train_X)  # .transpose((0, 2, 1)).reshape(1, -1, 1)
            X.append(t_X)
        if len(set(x.shape for x in X)) == 1:
            # same shape
            X = np.concatenate(X, axis=0)
        else:
            X = np.concatenate(X, axis=1)
        self.model.fit(X, *args, **kwargs)

    def evaluate(self, dataset, seq_len=96, pred_len=96):
        all_metrics = []
        if not isinstance(dataset, list):
            dataset = [dataset]
        for d, transform in zip(dataset, self.transform):
            if d.type == 'm4':
                test_X = d.train_data
                test_Y = d.test_data
                pred_len = d.test_data.shape[-1]
            else:
                # test_data: (num_series, num_samples, num_features)
                test_data = d.test_data
                test_data = transform.transform(test_data)
                subseries = np.concatenate(
                    ([sliding_window_view(v, (seq_len + pred_len, v.shape[-1])) for v in test_data]))
                # subseries: (num_samples, num_series, seq_len + pred_len, num_features)
                test_X = subseries[:, 0, :seq_len, :]
                test_Y = subseries[:, 0, seq_len:, :]
            te_X = test_X
            fore = self.model.forecast(te_X, pred_len=pred_len)
            all_metrics.append(metrics(fore, test_Y))
        return all_metrics[0] if len(all_metrics) == 1 else all_metrics
