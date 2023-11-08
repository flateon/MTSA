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

    trend_a = np.zeros_like(X)
    seasonal_a = np.zeros_like(X)
    filter = np.repeat(1.0 / period, period)

    for i, x in enumerate(X):
        x_pad = np.pad(x, ((period - 1) // 2, (period) // 2), mode='wrap')
        trend = np.convolve(x_pad, filter, mode='valid')

        seasonal = x - trend
        #
        # period_row = detrended[:timestamp // period * period].reshape((-1, period))
        # period_averages = np.mean(period_row, axis=0)
        #
        # seasonal = np.tile(period_averages, timestamp // period + 1)[:timestamp]

        trend_a[i], seasonal_a[i] = trend, seasonal
    trend_a = trend_a.reshape((n_samples, n_channels, timestamp)).transpose((0, 2, 1))
    seasonal_a = seasonal_a.reshape((n_samples, n_channels, timestamp)).transpose((0, 2, 1))
    return trend_a, seasonal_a


def differential_decomposition(x):
    """
    Differential Decomposition Algorithm
    Args:
        x (numpy.ndarray): Input time series data
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """

    raise NotImplementedError
