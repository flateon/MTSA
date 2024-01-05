import matplotlib.pyplot as plt
import numpy as np
from src.utils.metrics import safe_divide


def pearson_correlation(X, y):
    """
    Calculate the Pearson correlation coefficient between each feature in X and the target variable y.

    Parameters:
    - X: ndarray of shape (n_samples, n_features): Input features.
    - y: ndarray of shape (n_samples, 1): Target variable.

    Returns:
    - correlations: ndarray of shape (n_features,): Pearson correlation coefficients for each feature.
    """
    X = X - np.mean(X, axis=0)
    y = y - np.mean(y)

    covariance = y.T.dot(X)

    x_std = np.linalg.norm(X, axis=0)
    y_std = np.linalg.norm(y)

    covariance = safe_divide(covariance, x_std)
    covariance = safe_divide(covariance, y_std)
    return covariance.squeeze()


def data_visualize(dataset, t):
    """
    Choose t continous time points in data and visualize the chosen points. Note that some datasets have more than one
    channel.
    param:
        dataset: dataset to visualize
        t: the number of timestamps to visualize
    """
    # multichannel dataset
    if len(dataset.train_data.shape) == 3:
        # data: [timestamp, channels]
        data = dataset.train_data[0, ...]
        col_name = dataset.data_cols
        data_len = data.shape[-2]
        t_start = np.random.randint(0, data_len - t)
        data = data[t_start:t_start + t, :]
        t = np.arange(t_start, t_start + t)

        X, Y = np.split(data, [-1], axis=-1)
        correlation = pearson_correlation(X, Y)
        # sort by correlation
        idx = np.argsort(correlation)[::-1]

        top_k_idx = idx[:3].tolist()
        bottom_k_idx = idx[-3:].tolist()
        # close_to_zero_idx = np.argsort(np.abs(correlation))[:2].tolist()

        fig, axes = plt.subplots(6, 1, dpi=300, figsize=(12, 16), constrained_layout=True)
        fig.suptitle(dataset.name, fontsize=18)

        for i, ax in zip(top_k_idx + bottom_k_idx, axes):
            ax.plot(t, X[..., i], label=col_name[i])
            ax.set_title(f'Feature {col_name[i]}\nCorrelation {correlation[i]:.3f}')
            ax.legend(loc='upper left')

            ax1 = ax.twinx()
            ax1.plot(t, Y, label='target', c='r')
            ax1.legend(loc='upper right')

    # single channel dataset
    else:
        # (n_samples, timesteps)
        data = dataset.test_data
        if len(data.shape) == 1:
            n_samples = len(data)
            data_len = len(data[0])
        else:
            n_samples, data_len = data.shape

        if n_samples >= 6:
            idx_samples = np.random.choice(n_samples, 6, replace=False)
        else:
            idx_samples = np.arange(n_samples)

        t_start = np.random.randint(0, data_len - t)
        data = data[idx_samples][t_start:t_start + t]
        t = np.arange(t_start, t_start + t)

        fig, axes = plt.subplots(len(idx_samples), 1, dpi=300, figsize=(12, 16), constrained_layout=True)
        if len(idx_samples) == 1:
            axes = [axes]
        fig.suptitle(dataset.name, fontsize=18)

        for d, ax, idx in zip(data, axes, idx_samples):
            ax.plot(t, d, label=f'sample: {idx}')
            ax.legend(loc='upper left')

    save_path = f'./imgs/{dataset.name}.png'
    plt.savefig(save_path)
    print(f'Image is saved in {save_path}')
    # plt.show()
