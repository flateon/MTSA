import argparse
import random
import time

import numpy as np

from main import get_model, get_transform
from src.dataset.dataset import get_dataset
from trainer import MLTrainer
import pandas as pd

from main import get_args

if __name__ == '__main__':

    args = get_args()
    print("| dataset | distance | n_neighbors | time | num_bits | num_hashes | mse | mae | mape | smape | mase |")
    results = []

    for bits in (8,):
        for hashes in (12, 14, 16, 18, 20):
            for n_neighbors in (21, 31, 41, 51, 71, 91):
                fix_seed = 2023
                random.seed(fix_seed)
                np.random.seed(fix_seed)

                args.num_bits = bits
                args.n_neighbors = n_neighbors
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
                trainer.train(args)
                # evaluate model
                start = time.time()
                mse, mae, mape, smape, mase = trainer.evaluate(dataset, seq_len=args.seq_len,
                                                               pred_len=args.pred_len)
                end = time.time()
                print(
                    f"| {dataset.name} | {distance} | {n_neighbors} | {end - start:.2f} | {bits} | {hashes} | {mse:.4g} | {mae:.4g} | {mape:.4g} | {smape:.4g} | {mase:.4g} |")

                results.append([bits, hashes, n_neighbors, end - start, mse, mae, mape, smape, mase])

    results_df = pd.DataFrame(results,
                              columns=["num_bits", "num_hashes", "n_neighbors", "time", "mse", "mae", "mape",
                                       "smape", "mase"])

    results_df.to_csv("benchmark_lsh.csv", index=False)
    results_df.to_csv("benchmark_lsh_3g.csv", index=False, float_format='%.3g')
