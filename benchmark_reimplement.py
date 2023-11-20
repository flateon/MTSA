import random

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from main import get_model, get_transform
from src.dataset.dataset import get_dataset, ETTDataset
from src.utils.metrics import metrics
from trainer import MLTrainer
import pandas as pd
from main import get_args

ALL_MODEL = (
    # 'ZeroForecast',
    # 'MeanForecast',
    # 'LinearRegression',
    # 'ExponentialSmoothing',
    # 'TsfKNN',
    'DLinear',
    # 'DLinearClosedForm',
)


def split_data(self):
    self.split = True
    if self.frequency == 'h':
        self.num_train = 12 * 30 * 24
        self.num_val = 4 * 30 * 24
        self.num_test = 4 * 30 * 24
    elif self.frequency == 'm':
        self.num_train = 18 * 30 * 24 * 4
        self.num_val = 5 * 30 * 24 * 4
    self.train_data = self.data[:, :self.num_train, :]
    self.val_data = self.data[:, self.num_train: self.num_train + self.num_val, :]
    self.test_data = self.data[:, self.num_train + self.num_val:self.num_train + self.num_val + self.num_test, :]


def no_invert_evaluate(self, dataset, seq_len=96, pred_len=32):
    if dataset.type == 'm4':
        test_X = dataset.train_data
        test_Y = dataset.test_data
        pred_len = dataset.test_data.shape[-1]
    else:
        # test_data: (num_series, num_samples, num_features)
        test_data = dataset.test_data
        subseries = np.concatenate(([sliding_window_view(v, (seq_len + pred_len, v.shape[-1])) for v in test_data]))
        # subseries: (num_samples, num_series, seq_len + pred_len, num_features)
        test_X = subseries[:, 0, :seq_len, :]
        test_Y = subseries[:, 0, seq_len:, :]
    te_X = self.transform.transform(test_X)
    fore = self.model.forecast(te_X, pred_len=pred_len)
    # fore = self.transform.inverse_transform(fore)
    test_Y = self.transform.transform(test_Y)
    return metrics(fore, test_Y)


if __name__ == '__main__':
    args = get_args()
    print(" | Model | mse | mae | mape | smape | mase |")
    results = []

    for model_name in ALL_MODEL:
        for invert_transform in (True, False):
            fix_seed = 2023
            random.seed(fix_seed)
            np.random.seed(fix_seed)

            args.model = model_name
            args.decomposition = 'moving_average'
            args.distance = 'decompose'
            args.seq_len = 336
            args.pred_len = 96
            model_name = "DLinear" if invert_transform else "DLinear no inverse transform"

            ETTDataset.split_data = split_data
            dataset = ETTDataset(args)
            # create model
            model = get_model(args)
            # data transform
            transform = get_transform(args)
            # create trainer
            if not invert_transform:
                MLTrainer.evaluate = no_invert_evaluate
            trainer = MLTrainer(model=model, transform=transform, dataset=dataset)
            # train model
            trainer.train(args)
            # evaluate model
            mse, mae, mape, smape, mase = trainer.evaluate(dataset, seq_len=args.seq_len,
                                                           pred_len=args.pred_len)
            print(
                f"| {model_name:20} | {mse:5.4g} | {mae:5.4g} | {mape:5.4g} | {smape:5.4g} | {mase:5.4g} |")
            results.append([model_name, mse, mae, mape, smape, mase])

# Create a Pandas DataFrame from the results list
results_df = pd.DataFrame(results, columns=["Model", "MSE", "MAE", "MAPE", "SMAPE", "MASE"])

# Save the DataFrame to a CSV file
results_df.to_csv("results/test_reimplement.csv", index=False)
results_df.to_csv("results/test_reimplement_3g.csv", index=False, float_format='%.3g')
