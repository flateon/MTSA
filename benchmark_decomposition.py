import random

import numpy as np

from main import get_model, get_transform
from src.dataset.dataset import get_dataset
from trainer import MLTrainer
import pandas as pd
from main import get_args

ALL_MODEL = (
    # 'ZeroForecast',
    # 'MeanForecast',
    # 'LinearRegression',
    # 'ExponentialSmoothing',
    'TsfKNN',
    'DLinear',
    'DLinearClosedForm',
)

ALL_DECOMPOSITION = (
    'moving_average',
    'differential',
    'classic',
)

if __name__ == '__main__':
    args = get_args()
    print(" | Model | Decomposition | mse | mae | mape | smape | mase |")
    results = []

    dataset = get_dataset(args)

    for model_name in ALL_MODEL:
        for decomposition in ALL_DECOMPOSITION:
            fix_seed = 2023
            random.seed(fix_seed)
            np.random.seed(fix_seed)

            args.model = model_name
            args.decomposition = decomposition
            args.distance = 'decompose'

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
                f"| {model_name:20}| {decomposition:20} | {mse:5.4g} | {mae:5.4g} | {mape:5.4g} | {smape:5.4g} | {mase:5.4g} |")
            results.append(
                [model_name, decomposition, mse, mae, mape, smape, mase])

    # Create a Pandas DataFrame from the results list
    results_df = pd.DataFrame(results,
                              columns=["Model", "Decomposition", "MSE", "MAE", "MAPE", "SMAPE",
                                       "MASE"])

    # Save the DataFrame to a CSV file
    results_df.to_csv("results/test_decompose.csv", index=False)
    results_df.to_csv("results/test_decompose_3g.csv", index=False, float_format='%.3g')
