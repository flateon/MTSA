import argparse
import random

import numpy as np

from main import get_model, get_transform
from src.dataset.dataset import get_dataset
from trainer import MLTrainer
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()

    # dataset config
    parser.add_argument('--data_path', type=str, default='./dataset/ETT/ETTh1.csv')
    parser.add_argument('--train_data_path', type=str, default='./dataset/m4/Daily-train.csv')
    parser.add_argument('--test_data_path', type=str, default='./dataset/m4/Daily-test.csv')
    parser.add_argument('--dataset', type=str, default='Custom', help='dataset type, options: [M4, ETT, Custom]')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--ratio_train', type=int, default=0.7, help='train dataset length')
    parser.add_argument('--ratio_val', type=int, default=0, help='validate dataset length')
    parser.add_argument('--ratio_test', type=int, default=0.3, help='input sequence length')

    # forcast task config
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=32, help='prediction sequence length')

    # model define
    parser.add_argument('--model', type=str, default='TsfKNN', help='model name')
    parser.add_argument('--lamda', type=float, default=1, help='lamda for Yeo Johnson Transform')
    parser.add_argument('--n_neighbors', type=int, default=51, help='number of neighbors used in TsfKNN')
    parser.add_argument('--distance', type=str, default='chebyshev', help='distance used in TsfKNN')
    parser.add_argument('--msas', type=str, default='MIMO', help='multi-step ahead strategy used in TsfKNN, options: '
                                                                 '[MIMO, recursive]')
    parser.add_argument('--knn', type=str, default='lsh', help='knn method used in TsfKNN, options: '
                                                               '[brute_force, lsh]')
    parser.add_argument('--num_bits', type=int, default=8, help='num of bits for lsh method used in TsfKNN')
    parser.add_argument('--num_hashes', type=int, default=1, help='num of hashes for lsh method used in TsfKNN')
    parser.add_argument('--ew', type=float, default=0.9, help='weight of Exponential Smoothing model')

    # transform define
    parser.add_argument('--transform', type=str, default='IdentityTransform')

    args = parser.parse_args()
    return args


ALL_DATASET = (
    ('./dataset/electricity/electricity.csv', {'lamda': 0.4686}),
    ('./dataset/exchange_rate/exchange_rate.csv', {'lamda': 2.8284}),
    ('./dataset/illness/national_illness.csv', {'lamda': 0.8973}),
    ('./dataset/traffic/traffic.csv', {'lamda': 3.5261}),
    ('./dataset/weather/weather.csv', {'lamda': 2.3604}),
    ('./dataset/ETT/ETTh1.csv', {'lamda': 0.5505}),
    ('./dataset/ETT/ETTh2.csv', {'lamda': 0.7854}),
    ('./dataset/ETT/ETTm1.csv', {'lamda': 0.5501}),
    ('./dataset/ETT/ETTm2.csv', {'lamda': 0.7837}),
)

ALL_TRANSFORM = (
    'IdentityTransform',
    'NormalizationTransform',
    'StandardizationTransform',
    'MeanNormalizationTransform',
    'YeoJohnsonTransform',
)

ALL_MODEL = (
    'ZeroForecast',
    'MeanForecast',
    'LinearRegression',
    'ExponentialSmoothing',
    # 'TsfKNN',
)

if __name__ == '__main__':
    args = get_args()
    print("| dataset | model | transform | mse | mae | mape | smape | mase |")
    results = []

    for dataset_path, kwargs in ALL_DATASET:
        args.data_path = dataset_path
        # update  args
        for k, v in kwargs.items():
            setattr(args, k, v)
        dataset = get_dataset(args)

        for model_name in ALL_MODEL:
            args.model = model_name
            for transform_name in ALL_TRANSFORM:
                args.transform = transform_name

                fix_seed = 2023
                random.seed(fix_seed)
                np.random.seed(fix_seed)
                # create model
                model = get_model(args)
                # data transform
                transform = get_transform(args)
                # create trainer
                trainer = MLTrainer(model=model, transform=transform, dataset=dataset)
                # train model
                trainer.train()
                # evaluate model
                mse, mae, mape, smape, mase = trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len)
                print(
                    f"| {dataset.name} | {model_name} | {transform_name.removesuffix('Transform')} | {mse:.4g} | {mae:.4g} | {mape:.4g} | {smape:.4g} | {mase:.4g} |")
                results.append(
                    [dataset.name, model_name, transform_name.removesuffix('Transform'), mse, mae, mape, smape, mase])

    # Create a Pandas DataFrame from the results list
    results_df = pd.DataFrame(results, columns=["dataset", "model", "transform", "mse", "mae", "mape", "smape", "mase"])

    # Save the DataFrame to a CSV file
    results_df.to_csv("model_metrics.csv", index=False)
