import numpy as np


def mse(predict, target):
    return np.mean((target - predict) ** 2)


def mae(predict, target):
    return np.mean(np.abs(target - predict))


def mape(predict, target):
    return 100 * np.mean(np.abs((target - predict) / target))


def smape(predict, target):
    return 200 * np.mean(np.abs(target - predict) / (np.abs(target) + np.abs(predict)))


def mase(predict, target, m: int):
    assert len(target) > m
    assert isinstance(m, int)
    return np.mean(np.abs(target - predict)) / np.mean(np.abs(target[m:] - target[:-m]))
