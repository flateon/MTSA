import random
import time

import numpy as np

from main import get_model, get_transform
from src.dataset.dataset import get_dataset
from trainer import MLTrainer
import pandas as pd
from main import get_args

ALL_DATASET = (
    # ('./dataset/illness/national_illness.csv', {'lamda': 0.8973, 'dataset': 'Custom', 'seq_len': 36}),
    # ('./dataset/ETT/ETTh1.csv', {'lamda': 0.5505, 'frequency': 'h', 'period': 24, 'dataset': 'ETT'}),
    ('./dataset/ETT/ETTh2.csv', {'lamda': 0.7854, 'frequency': 'h', 'period': 24, 'dataset': 'ETT'}),
    ('./dataset/ETT/ETTm1.csv', {'lamda': 0.5501, 'frequency': 'm', 'period': 96, 'dataset': 'ETT'}),
    ('./dataset/ETT/ETTm2.csv', {'lamda': 0.7837, 'frequency': 'm', 'period': 96, 'dataset': 'ETT'}),
    # ('./dataset/exchange_rate/exchange_rate.csv', {'lamda': 2.8284, 'dataset': 'Custom', 'period': 24}),
    # ('./dataset/weather/weather.csv', {'lamda': 2.3604, 'dataset': 'Custom', 'period': 24}),
    # ('./dataset/electricity/electricity.csv', {'lamda': 0.4686, 'dataset': 'Custom', 'period': 24}),  # OOM
    # ('./dataset/traffic/traffic.csv', {'lamda': 3.5261, 'dataset': 'Custom'}),  # OOM
)

PRED_LEN = (
    96,
    # 192,
    # 336,
    # 720,
)

ALL_COMPONENTS = (
    #'mle',
    '0.1',
    '0.3',
    '0.5',
    '0.7',
    '0.9',
    '0.95',
    '0.99',
    None,  # all
)

if __name__ == '__main__':
    args = get_args()
    print("| dataset  | model                | pred | mse    | mae    | mape   | smape  | mase   | time |")
    results = []
    for dataset_path, kwargs in ALL_DATASET:
        args.data_path = dataset_path
        # update  args
        for k, v in kwargs.items():
            setattr(args, k, v)
        dataset = get_dataset(args)
        args.model = 'SPIRIT'

        for components in ALL_COMPONENTS:
            args.pca_components = components
            for pred_len in PRED_LEN:
                args.pred_len = pred_len

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
                start = time.time()
                trainer.train(args)
                # evaluate model
                mse, mae, mape, smape, mase = trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len)
                end = time.time()
                try:
                    print(
                        f"| {dataset.name:8} | {components:20} | {pred_len:3} | {mse:6.4g} | {mae:6.4g} | {mape:6.4g} | {smape:6.4g} | {mase:6.4g} | {end - start: 6.3g} |")
                except:
                    pass
                results.append(
                    [dataset.name, components, model.n_components, pred_len, mse, mae, mape, smape, mase,
                     end - start])
            # Create a Pandas DataFrame from the results list
            results_df = pd.DataFrame(results,
                                      columns=["dataset", "components", "n_components", "pred_len", "mse", "mae",
                                               "mape",
                                               "smape", "mase", "time"])

            # Save the DataFrame to a CSV file
            results_df.to_csv("results/SPIRIT.csv", index=False)
