import numpy as np


def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b != 0)


def mse(predict, target):
    return np.mean((target - predict) ** 2)


def mae(predict, target):
    return np.mean(np.abs(target - predict))


def mape(predict, target):
    # fix divide by zero
    return 100 * np.mean(np.abs(safe_divide((target - predict), target)))


def smape(predict, target):
    return 200 * np.mean(safe_divide(np.abs(target - predict), (np.abs(target) + np.abs(predict))))


def mase(predict, target, m: int = 24):
    mase_ = np.mean(np.abs(target - predict)) / np.mean(np.abs(target[m:] - target[:-m]))
    return np.nan_to_num(mase_, copy=False, nan=0, posinf=0.0, neginf=0.0)


def metrics(pred, target, m: int = 24):
    return mse(pred, target), mae(pred, target), mape(pred, target), smape(pred, target), mase(pred, target, m=m)
