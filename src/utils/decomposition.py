from typing import Union

import numpy as np
from scipy.ndimage import convolve1d
from scipy.optimize import minimize
from statsmodels.tsa.stl._stl import STL


def moving_average(x, period: Union[int, tuple] = 24, return_seasonal: bool = True):
    """
    Moving Average Algorithm
    Args:
        x (numpy.ndarray): Input time series data
        period (int or tuple): Seasonal period
        return_seasonal (bool): Whether to return seasonal component
    Returns:
        trend (numpy.ndarray): Trend component
        detrend (numpy.ndarray): Detrended time series
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """
    if isinstance(period, int):
        period = (period, 1)

    weights = np.zeros(sum(period) - 1)
    for i in range(period[1]):
        weights[i: i + period[0]] += 1
    weights /= weights.sum()

    trend = convolve1d(x, weights, axis=1, mode='nearest')

    if not return_seasonal:
        return trend
    else:
        seasonal = x - trend
        return trend, seasonal


def get_henderson_weights(period: int = 13):
    """
    Get Henderson weights
    Args:
        period (int): Seasonal period
    Returns:
        weights (numpy.ndarray): Henderson weights
    """
    assert period % 2 == 1, "Period must be odd"
    n = period // 2 + 2
    weights = [315 * ((n - 1) ** 2 - i ** 2) * (n ** 2 - i ** 2) * ((n + 1) ** 2 - i ** 2) * (
            3 * n ** 2 - 16 - 11 * i ** 2) / (
                       8 * n * (n ** 2 - 1) * (4 * n ** 2 - 1) * (4 * n ** 2 - 9) * (4 * n ** 2 - 25)) for i in
               range(-period // 2 + 1, period // 2 + 1)]
    return weights


def get_henderson_weights2(period: int = 13):
    """
    Get Henderson weights
    Args:
        period (int): Seasonal period
    Returns:
        weights (numpy.ndarray): Henderson weights
    """
    assert period % 2 == 1, "Period must be odd"
    func = lambda x: (np.diff(x, n=3) ** 2).sum()
    cons = (
        {'type': 'eq', 'fun': lambda x: x.sum() - 1},
        {'type': 'eq', 'fun': lambda x: (x * (np.arange(period) - period // 2)).sum()},
        {'type': 'eq', 'fun': lambda x: (x * (np.arange(period) - period // 2) ** 2).sum()},
    )
    # x0 = np.random.randn(period)
    x0 = np.ones(period) / period
    res = minimize(func, x0, constraints=cons)
    return res.x


def henderson_moving_average(x, period: int = 13):
    """
    Henderson Moving Average Algorithm
    Args:
        x (numpy.ndarray): Input time series data
        period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component
    """
    weights = get_henderson_weights(period)

    trend = convolve1d(x, weights, axis=1, mode='nearest')
    return trend


def differential_decomposition(x, *args, **kwargs):
    """
    Differential Decomposition Algorithm
    Args:
        x (numpy.ndarray): Input time series data (n_samples, timestamp, n_channels)
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """
    seasonal = np.pad(np.diff(x, axis=1), ((0, 0), (1, 0), (0, 0)), mode='edge')
    trend = x - seasonal
    return trend, seasonal


def classic_decomposition(x, period=24):
    """
    Classic Decomposition Algorithm
    Args:
        x (numpy.ndarray): Input time series data (n_samples, timestamp, n_channels)
        period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
        resid (numpy.ndarray): Residual component
    """
    n_samples, timestamp, n_channels = x.shape
    n_periods = timestamp // period

    trend, detrend = moving_average(x, period)
    period_avg = detrend[:, :n_periods * period, :].reshape((n_samples, n_periods, period, n_channels)).mean(axis=1)

    seasonal = np.tile(period_avg, (1, n_periods + 1, 1))[:, :timestamp, :]

    resid = detrend - seasonal
    return trend, seasonal, resid


def STL_decomposition(x, seasonal_period=13):
    """
    Seasonal and Trend decomposition using Loess
    Args:
        x (numpy.ndarray): Input time series data
        seasonal_period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
        residual (numpy.ndarray): Residual component
    """
    trend, seasonal, residual = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    for n in range(x.shape[0]):
        for c in range(x.shape[2]):
            result = STL(x[n, :, c], seasonal_period).fit()
            trend[n, :, c], seasonal[n, :, c], residual[n, :, c] = result.trend, result.seasonal, result.resid
    return trend, seasonal, residual


def X11_decomposition(x, seasonal_period=13):
    """
    X11 decomposition
    Args:
        x (numpy.ndarray): Input time series data
        seasonal_period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
        residual (numpy.ndarray): Residual component
    """
    # T_1 = M_2x12(x)
    T_1 = moving_average(x, (2, 12), return_seasonal=False)
    Y_1 = x / T_1
    S_1 = moving_average(Y_1, (3, 3), False) / moving_average(Y_1, (2, 12), False)
    X_2 = x / S_1

    T_2 = henderson_moving_average(X_2, period=13)
    Y_2 = x / T_2
    S_2 = moving_average(Y_2, (3, 5), False) / moving_average(Y_2, (2, 12), False)
    X_3 = x / S_2

    T_3 = henderson_moving_average(X_3, period=seasonal_period)
    I_3 = X_3 / T_3

    return T_3, S_2, I_3


def get_decomposition(algorithm):
    """
    Get decomposition algorithm
    Args:
        algorithm (str): Decomposition algorithm
    Returns:
        decomposition_fn (function): Decomposition function
        n_args (int): Number of components
    """
    if algorithm == 'moving_average':
        return moving_average, 2
    elif algorithm == 'differential':
        return differential_decomposition, 2
    elif algorithm == 'classic':
        return classic_decomposition, 3
    elif algorithm == 'henderson':
        return henderson_moving_average, 1
    elif algorithm == 'STL':
        return STL_decomposition, 3
    elif algorithm == 'X11':
        return X11_decomposition, 3
    else:
        raise ValueError('Algorithm {} is not supported.'.format(algorithm))
