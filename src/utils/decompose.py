import numpy as np


def seasonal_decompose(X: np.ndarray, period: int = 24) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Decompose a time series into its seasonal and trend components.

    :param X: Time series data shape=(n_samples, n_features).
    :param period: Period of the seasonal component.
    :return: Tuple of seasonal, trend, and residual components.
    """
    trend_a = np.zeros_like(X)
    seasonal_a = np.zeros_like(X)
    resid_a = np.zeros_like(X)

    for i, x in enumerate(X):
        n_samples = len(x)

        filt = np.repeat(1.0 / period, period)

        trend = np.convolve(x, filt, mode='same')

        detrended = x - trend

        period_row = detrended[:n_samples // period * period].reshape((-1, period))
        period_averages = np.mean(period_row, axis=0)

        seasonal = np.tile(period_averages, n_samples // period + 1)[:n_samples]

        resid = detrended - seasonal

        trend_a[i], seasonal_a[i], resid_a[i] = trend, seasonal, resid
    return trend_a, seasonal_a, resid_a


# # Example usage
# if __name__ == '__main__':
#     np.random.seed(0)
#     time = np.arange(0, 100, 1)
#
#     seasonal = 30 * np.sin(2 * np.pi * time / 12)
#     trend = np.arange(0, len(time), 1)
#     resid = np.random.normal(0, 1, len(time))
#
#     data = trend + seasonal + resid
#
#     seasonal, trend, resid = seasonal_decompose(data[np.newaxis, :], period=12)
#
#     # Plot the components
#     import matplotlib.pyplot as plt
#
#     plt.figure(figsize=(12, 6))
#     plt.subplot(4, 1, 1)
#     plt.plot(time, data, label='Original Data')
#     plt.legend()
#
#     plt.subplot(4, 1, 2)
#     plt.plot(time, seasonal[0], label='Seasonal Component')
#     plt.legend()
#
#     plt.subplot(4, 1, 3)
#     plt.plot(time, trend[0], label='Trend Component')
#     plt.legend()
#
#     plt.subplot(4, 1, 4)
#     plt.plot(time, resid[0], label='Residual Component')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
