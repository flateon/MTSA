import numpy as np

np.seterr(divide='ignore')


def mse(predict, target):
    return np.mean((target - predict) ** 2)


def mae(predict, target):
    return np.mean(np.abs(target - predict))


def mape(predict, target):
    ape = np.abs((target - predict) / target)
    # fix divide by zero
    return 100 * np.mean(np.nan_to_num(ape, copy=False, nan=0.0, posinf=0.0, neginf=0.0))


def smape(predict, target):
    sape = np.abs(target - predict) / (np.abs(target) + np.abs(predict))
    return 200 * np.mean(np.nan_to_num(sape, copy=False, nan=0.0, posinf=0.0, neginf=0.0))


def mase(predict, target, m: int):
    mase_ = np.mean(np.abs(target - predict)) / np.mean(np.abs(target[m:] - target[:-m]))
    return np.nan_to_num(mase_, copy=False, nan=0, posinf=0.0, neginf=0.0)
