import random
import time
from pathlib import Path

import numpy as np
import torch

from main import get_model, get_transform
from src.dataset.dataset import get_dataset
from trainer import MLTrainer
import pandas as pd
from main import get_args

ARGS_CONFIG = {
    'batch_size': 128,
}

ALL_DATASET = (
    # ('./dataset/traffic/traffic.csv', {'lamda': 3.5261, 'dataset': 'Custom'}),  # OOM
    # ('./dataset/electricity/electricity.csv', {'lamda': 0.4686}),  # OOM
    # ('./dataset/illness/national_illness.csv', {'lamda': 0.8973, 'dataset': 'Custom', 'seq_len': 36}),
    ('./dataset/weather/weather.csv', {'lamda': 2.3604, 'dataset': 'Custom', 'period': 24}),
    ('./dataset/exchange_rate/exchange_rate.csv', {'lamda': 2.8284, 'dataset': 'Custom', 'period': 24}),
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
    for k, v in ARGS_CONFIG.items():
        setattr(args, k, v)

    results = []

    for dataset_path, kwargs in ALL_DATASET:
        args.data_path = dataset_path
        # update  args
        for k, v in kwargs.items():
            setattr(args, k, v)
        dataset = get_dataset(args)

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
                transform.transform(dataset.train_data)

                pretrain = torch.load(f'checkpoints/{args.model}_{args.pred_len}/pretrain.pth')
                finetune = torch.load(f'checkpoints/{args.model}_{args.pred_len}/{dataset.name}_lp.pth')
                assert set(pretrain.keys()) == set(finetune.keys())

                for alpha in np.linspace(0, 1, 11):
                    weights = {key: (1 - alpha) * pretrain[key] + alpha * finetune[key] for key in pretrain.keys()}
                    model.model.load_state_dict(weights)
                    model.fitted = True

                    # create trainer
                    trainer = MLTrainer(model=model, transform=transform, dataset=dataset)

                    # evaluate model
                    mse, mae, _, _, _ = trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len,
                                                         mode='test')
                    val_mse, val_mae, _, _, _ = trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len,
                                                                 mode='val')
                    print(
                        f"| {alpha:.3f} | {dataset.name:15} | {model_name:20} | {pred_len:3} | {mse:6.4g} | {mae:6.4g} |")
                    results.append(
                        [alpha, dataset.name, model_name, pred_len, mse, mae, val_mse, val_mae])

    # Create a Pandas DataFrame from the results list
    results_df = pd.DataFrame(results,
                              columns=["alpha", "dataset", "model", "pred_len", "mse", "mae", "val_mse", "val_mae"])

    # Save the DataFrame to a CSV file
    results_df.to_csv("results/ensemble_lp.csv", index=False)
