import random
import time
from pathlib import Path

import numpy as np
import torch

from main import get_model, get_transform
from src.dataset.dataset import get_dataset, MultiDataset
from src.utils.transforms import ListTransform
from trainer import MLTrainer
import pandas as pd
from main import get_args

ALL_DATASET = (
    # ('./dataset/traffic/traffic.csv', {'lamda': 3.5261, 'dataset': 'Custom'}),  # OOM
    # ('./dataset/electricity/electricity.csv', {'lamda': 0.4686, 'dataset': 'Custom', 'period': 24}),  # OOM
    # ('./dataset/illness/national_illness.csv', {'lamda': 0.8973, 'dataset': 'Custom', 'seq_len': 36}),
    # ('./dataset/weather/weather.csv', {'lamda': 2.3604, 'dataset': 'Custom', 'period': 24}),
    # ('./dataset/exchange_rate/exchange_rate.csv', {'lamda': 2.8284, 'dataset': 'Custom', 'period': 24}),
    ('./dataset/ETT/ETTm1.csv', {'lamda': 0.5501, 'frequency': 'm', 'period': 96, 'dataset': 'ETT'}),
    ('./dataset/ETT/ETTm2.csv', {'lamda': 0.7837, 'frequency': 'm', 'period': 96, 'dataset': 'ETT'}),
    ('./dataset/ETT/ETTh1.csv', {'lamda': 0.5505, 'frequency': 'h', 'period': 24, 'dataset': 'ETT'}),
    ('./dataset/ETT/ETTh2.csv', {'lamda': 0.7854, 'frequency': 'h', 'period': 24, 'dataset': 'ETT'}),
)

PRED_LEN = (
    96,
    192,
    336,
    720,
)

ALL_MODEL = (
    # 'DLinear',
    # 'FLinearGD',
    'PatchTST',
    # 'Transformer',
)

if __name__ == '__main__':
    args = get_args()
    results = []

    all_dataset = []
    for dataset_path, kwargs in ALL_DATASET:
        args.data_path = dataset_path
        # update  args
        for k, v in kwargs.items():
            setattr(args, k, v)
        all_dataset.append(get_dataset(args))
    multi_dataset = MultiDataset(args)
    multi_dataset.merge_dataset(all_dataset)
    args.dataset = 'MULTI'

    for model_name in ALL_MODEL:
        args.model = model_name
        for pred_len in PRED_LEN:
            args.pred_len = pred_len

            fix_seed = 2023
            random.seed(fix_seed)
            np.random.seed(fix_seed)
            # create model
            model = get_model(args)
            # data transform
            transform = get_transform(args)
            list_transform = ListTransform(transform)
            # create trainer
            trainer = MLTrainer(model=model, transform=list_transform, dataset=multi_dataset)
            # train model
            trainer.train()

            save_path = Path(f'checkpoints/{args.model}_{args.pred_len}')
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.model.state_dict(), save_path / 'pretrain.pth')

            for dataset, transform in zip(all_dataset, list_transform.transforms):
                trainer = MLTrainer(model=model, transform=transform, dataset=dataset)
                mse, mae, _, _, _ = trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len)
                print(
                    f"| {dataset.name:15} | {model_name:20} | {pred_len:3} | {mse:6.4g} | {mae:6.4g} |")
                results.append(
                    [multi_dataset.name, dataset.name, model_name, pred_len, mse, mae])
            # Create a Pandas DataFrame from the results list
            results_df = pd.DataFrame(results,
                                      columns=["train_dataset", "test_dataset", "model", "pred_len", "mse", "mae",])

            # Save the DataFrame to a CSV file
            results_df.to_csv("results/pretrain_model.csv", index=False)
