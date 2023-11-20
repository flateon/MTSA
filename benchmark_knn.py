import random

import numpy as np

from main import get_model, get_transform
from src.dataset.dataset import get_dataset
from trainer import MLTrainer
import pandas as pd
from main import get_args

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
    results_df.to_csv("results/test_knn.csv", index=False)
    results_df.to_csv("results/test_knn_3g.csv", index=False, float_format='%.3g')
