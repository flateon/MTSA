import numpy as np


def moving_average(X, period=25):
    """
    Moving Average Algorithm
    Args:
        X (numpy.ndarray): Input time series data (n_samples, timestamp, n_channels)
        period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """
    n_samples, timestamp, n_channels = X.shape
    X = X.transpose((0, 2, 1)).reshape((-1, timestamp))

    trend = np.zeros_like(X)
    filter = np.repeat(1.0 / period, period)

    for i, x in enumerate(X):
        # x_pad = np.pad(x, ((period - 1) // 2, (period) // 2), mode='wrap')
        trend[i] = np.convolve(x, filter, mode='same')
    seasonal = X - trend
    trend = trend.reshape((n_samples, n_channels, timestamp)).transpose((0, 2, 1))
    seasonal = seasonal.reshape((n_samples, n_channels, timestamp)).transpose((0, 2, 1))
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
    seasonal = np.diff(x, axis=1)
    trend = x[:, 1:] - seasonal
    return trend, seasonal

