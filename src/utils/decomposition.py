import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def moving_average(x, period=24):
    """
    Moving Average Algorithm
    Args:
        x (numpy.ndarray): Input time series data (n_samples, timestamp, n_channels)
        period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """
    pad_width = np.array(((0, 0),
                          ((period - 1) // 2, period // 2),
                          (0, 0)))
    x_pad = np.pad(x, pad_width, mode='edge')
    trend = sliding_window_view(x_pad, period, axis=1).mean(axis=3)
    seasonal = x - trend
    return trend, seasonal


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
    else:
        raise ValueError('Algorithm {} is not supported.'.format(algorithm))
