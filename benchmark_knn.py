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
    parser.add_argument('--dataset', type=str, default='ETT', help='dataset type, options: [M4, ETT, Custom]')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--ratio_train', type=int, default=0.7, help='train dataset length')
    parser.add_argument('--ratio_val', type=int, default=0, help='validate dataset length')
    parser.add_argument('--ratio_test', type=int, default=0.3, help='input sequence length')
    parser.add_argument('--frequency', type=str, default='h', help='frequency of time series data, options: [h, m]')

    # forcast task config
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=32, help='prediction sequence length')

    # model define
    parser.add_argument('--model', type=str, default='TsfKNN', help='model name')
    parser.add_argument('--lamda', type=float, default=1, help='lamda for Yeo Johnson Transform')
    parser.add_argument('--n_neighbors', type=int, default=71, help='number of neighbors used in TsfKNN')
    parser.add_argument('--distance', type=str, default='chebyshev', help='distance used in TsfKNN')
    parser.add_argument('--msas', type=str, default='MIMO', help='multi-step ahead strategy used in TsfKNN, options: '
                                                                 '[MIMO, recursive]')
    parser.add_argument('--embedding', type=str, default='fourier', help='embedding method used in TsfKNN, options: '
                                                                         '[lag, fourier]')
    parser.add_argument('--tau', type=int, default=1, help='tau for lag embedding method used in TsfKNN')
    parser.add_argument('--knn', type=str, default='lsh', help='knn method used in TsfKNN, options: '
                                                               '[brute_force, lsh]')
    parser.add_argument('--num_bits', type=int, default=8, help='num of bits for lsh method used in TsfKNN')
    parser.add_argument('--num_hashes', type=int, default=16, help='num of hashes for lsh method used in TsfKNN')
    parser.add_argument('--ew', type=float, default=0.9, help='weight of Exponential Smoothing model')

    parser.add_argument('--individual', action='store_true', default=False)

    # transform define
    parser.add_argument('--transform', type=str, default='StandardizationTransform')

    args = parser.parse_args()
    return args


ALL_DISTANCE = (
    'euclidean',
    'manhattan',
    'chebyshev',
    # # 'minkowski',
    'cosine',
    'decompose',
    # 'zero',
)

ALL_EMBEDDING = (
    ('lag', 1),
    ('lag', 2),
    ('lag', 3),
    ('lag', 4),
    ('fourier', None),
)

if __name__ == '__main__':
    args = get_args()
    print(" | Embedding | Distance | mse | mae | mape | smape | mase |")
    results = []

    dataset = get_dataset(args)

    for embedding, tau in ALL_EMBEDDING:
        args.embedding = embedding
        args.tau = tau
        embedding = f'{embedding}(tau={tau})' if embedding == 'lag' else embedding

        for distance in ALL_DISTANCE:
            fix_seed = 2023
            random.seed(fix_seed)
            np.random.seed(fix_seed)

            args.distance = distance

            # create model
            model = get_model(args)
            # data transform
            transform = get_transform(args)
            # create trainer
            trainer = MLTrainer(model=model, transform=transform, dataset=dataset)
            # train model
            trainer.train(args)
            # evaluate model
            mse, mae, mape, smape, mase = trainer.evaluate(dataset, seq_len=args.seq_len,
                                                           pred_len=args.pred_len)
            print(
                f"| {embedding:12}| {distance:10} | {mse:5.4g} | {mae:5.4g} | {mape:5.4g} | {smape:5.4g} | {mase:5.4g} |")
            results.append(
                [embedding, distance, mse, mae, mape, smape, mase])

    # Create a Pandas DataFrame from the results list
    results_df = pd.DataFrame(results,
                              columns=["Embedding", "distance", "mse", "mae", "mape", "smape",
                                       "mase"])

    # Save the DataFrame to a CSV file
    results_df.to_csv("model_metrics_knn.csv", index=False)
    results_df.to_csv("model_metrics_knn_3g.csv", index=False, float_format='%.3g')
