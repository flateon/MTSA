import argparse
import random

import numpy as np
import torch

from src.dataset.dataset import get_dataset
from src.models.PatchTST import PatchTST
from src.models.Transformer import Transformer

from src.models.ARIMA import ARIMA
from src.models.DLinear import DLinear, DLinearClosedForm
from src.models.ResidualModel import FLinear, FLinearGD
from src.models.SPIRIT import SPIRIT
from src.models.ThetaMethod import ThetaMethod
from src.models.TsfKNN import TsfKNN
from src.models.baselines import LinearRegression, ExponentialSmoothing
from src.models.baselines import ZeroForecast, MeanForecast
from src.utils.transforms import IdentityTransform, StandardizationTransform
from src.utils.transforms import NormalizationTransform, MeanNormalizationTransform, YeoJohnsonTransform
from trainer import MLTrainer


def get_args():
    parser = argparse.ArgumentParser()

    # dataset config
    parser.add_argument('--data_path', type=str, default='./dataset/ETT/ETTh1.csv')
    parser.add_argument('--train_data_path', type=str, default='./dataset/m4/Daily-train.csv')
    parser.add_argument('--test_data_path', type=str, default='./dataset/m4/Daily-test.csv')
    parser.add_argument('--dataset', type=str, default='ETT', help='dataset type, options: [M4, ETT, Custom]')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--frequency', type=str, default='h', help='frequency of time series data, options: [h, m]')
    parser.add_argument('--ratio_train', type=float, default=0.7, help='ratio of training set')
    parser.add_argument('--ratio_val', type=float, default=0.1, help='ratio of validation set')
    parser.add_argument('--ratio_test', type=float, default=0.2, help='ratio of test set')
    parser.add_argument('--period', type=int, default=24, help='period of seasonal data')

    # forcast task config
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length in [96, 192, 336, 720]')

    # model define
    parser.add_argument('--model', type=str, default='TsfKNN', help='model name')
    parser.add_argument('--lamda', type=float, default=1, help='lamda for Yeo Johnson Transform')
    parser.add_argument('--n_neighbors', type=int, default=51, help='number of neighbors used in TsfKNN')
    parser.add_argument('--distance', type=str, default='cosine', help='distance used in TsfKNN')
    parser.add_argument('--msas', type=str, default='MIMO', help='multi-step ahead strategy used in TsfKNN, options: '
                                                                 '[MIMO, recursive]')
    parser.add_argument('--embedding', type=str, default='lag', help='embedding method used in TsfKNN, options: '
                                                                     '[lag, fourier]')
    parser.add_argument('--tau', type=int, default=2, help='tau for lag embedding method used in TsfKNN')
    parser.add_argument('--knn', type=str, default='lsh', help='knn method used in TsfKNN, options: [brute_force, lsh]')
    parser.add_argument('--num_bits', type=int, default=8, help='num of bits for lsh method used in TsfKNN')
    parser.add_argument('--num_hashes', type=int, default=1, help='num of hashes for lsh method used in TsfKNN')
    parser.add_argument('--ew', type=float, default=0.9, help='weight of Exponential Smoothing model')
    parser.add_argument('--fl_weight', type=str, default='linear', help='weight of FLinear model, options: [time, '
                                                                        'freq, linear, constant, learn]')
    parser.add_argument('--pca_components', type=str, default='0.99',
                        help='number of components for PCA, options: ["mle", int, float]')
    parser.add_argument('--individual', action='store_true', default=False)
    parser.add_argument('--decomposition', type=str, default='classic',
                        help='decomposition method, options: [moving_average, differential, classic]')

    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, \
                            b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='hidden size')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--n_heads', type=int, default=8, help='number of heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--output_attention', type=bool, default=False, help='output attention')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--patch_len', type=int, default=16, help='patch_len')

    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # gpu define
    parser.add_argument('--device', type=str, default='0', help='gpu id or cpu')

    # transform define
    parser.add_argument('--transform', type=str, default='StandardizationTransform')

    args = parser.parse_args()
    return args


def get_model(args):
    model_dict = {
        'ZeroForecast':         ZeroForecast,
        'MeanForecast':         MeanForecast,
        'Linear':               LinearRegression,
        'ExponentialSmoothing': ExponentialSmoothing,
        'TsfKNN':               TsfKNN,
        'DLinear':              DLinear,
        'DLinearClosedForm':    DLinearClosedForm,
        'ARIMA':                ARIMA,
        'Theta':                ThetaMethod,
        'FLinear':              FLinear,
        'FLinearGD':            FLinearGD,
        'SPIRIT':               SPIRIT,
        'PatchTST':             PatchTST,
        'Transformer':          Transformer,
    }
    return model_dict[args.model](args)


def get_transform(args):
    transform_dict = {
        'IdentityTransform':          IdentityTransform,
        'NormalizationTransform':     NormalizationTransform,
        'StandardizationTransform':   StandardizationTransform,
        'MeanNormalizationTransform': MeanNormalizationTransform,
        'YeoJohnsonTransform':        YeoJohnsonTransform,
    }
    return transform_dict[args.transform](args)


if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)

    args = get_args()
    # load dataset
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
    mse, mae = trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len)
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
