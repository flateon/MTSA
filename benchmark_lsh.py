import argparse
import random
import time

import numpy as np

from main import get_model, get_transform
from src.dataset.dataset import get_dataset
from trainer import MLTrainer
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()

    # dataset config
    parser.add_argument('--data_path', type=str, default='./dataset/ETT/ETTh1.csv')
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
    parser.add_argument('--lamda', type=float, default=0.5505, help='lamda for Yeo Johnson Transform')
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
    parser.add_argument('--transform', type=str, default='YeoJohnsonTransform')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    print("| dataset | distance | time | num_bits | num_hashes | mse | mae | mape | smape | mase |")
    results = []

    for bits in (1, 2, 4, 8):
        args.num_bits = bits
        for hashes in (1, 2, 4, 8):
            fix_seed = 2023
            random.seed(fix_seed)
            np.random.seed(fix_seed)

            args.num_hashes = hashes
            model_name = args.model
            transform_name = args.transform
            distance = args.distance
            # get dataset
            dataset = get_dataset(args)
            # create model
            model = get_model(args)
            # data transform
            transform = get_transform(args)
            # create trainer
            trainer = MLTrainer(model=model, transform=transform, dataset=dataset)
            # train model
            trainer.train()
            # evaluate model
            start = time.time()
            mse, mae, mape, smape, mase = trainer.evaluate(dataset, seq_len=args.seq_len,
                                                           pred_len=args.pred_len)
            end = time.time()
            print(
                f"| {dataset.name} | {distance} | {end - start:.2f} | {bits} | {hashes} | {mse:.4g} | {mae:.4g} | {mape:.4g} | {smape:.4g} | {mase:.4g} |")

            results.append([bits, hashes, end - start, mse, mae, mape, smape, mase])

    results_df = pd.DataFrame(results,
                              columns=["num_bits", "num_hashes", "time", "mse", "mae", "mape",
                                       "smape", "mase"])

    results_df.to_csv("benchmark_lsh.csv", index=False)
    results_df.to_csv("benchmark_lsh_3g.csv", index=False, float_format='%.3g')
