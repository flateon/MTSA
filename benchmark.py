import argparse
import random

import numpy as np
from sklearn.preprocessing import PowerTransformer

from main import get_model, get_transform, get_args
from src.dataset.dataset import get_dataset
from trainer import MLTrainer
import pandas as pd

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
    'TsfKNN',
)

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    np.random.seed(fix_seed)

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
