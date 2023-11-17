import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def moving_average(x, period=25):
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


def differential_decomposition(x):
    """
    Differential Decomposition Algorithm
    Args:
        x (numpy.ndarray): Input time series data (n_samples, timestamp, n_channels)
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """
    seasonal = np.pad(np.diff(x, axis=1), (1, 0), mode='edge')
    trend = x - seasonal
    return trend, seasonal
