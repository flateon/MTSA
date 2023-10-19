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

    # data: [timestamp, channels]
    data = dataset.train_data[0, ...]
    col_name = dataset.data_cols
    data_len = data.shape[-2]
    t_start = np.random.randint(0, data_len - t)
    data = data[t_start:t_start + t, :]

    X, Y = np.split(data, [-1], axis=-1)
    correlation = pearson_correlation(X, Y)
    # sort by correlation
    idx = np.argsort(correlation)

    bottom_k_idx = idx[:3]
    top_k_idx = idx[::-1][:3]

    fig, axes = plt.subplots(3, 2, dpi=300, figsize=(12, 16), constrained_layout=True)
    fig.suptitle(dataset.name, fontsize=18)

    for i, ax in zip(top_k_idx, axes[:, 0]):
        ax.plot(X[..., i], label=col_name[i])
        ax.set_title(f'Feature {col_name[i]}\nCorrelation {correlation[i]:.3f}')
        ax.legend(loc='upper left')

        ax1 = ax.twinx()
        ax1.plot(Y, label='target', c='r')
        ax1.legend(loc='upper right')

    for i, ax in zip(bottom_k_idx, axes[:, 1]):
        ax.plot(X[..., i], label=col_name[i])
        ax.set_title(f'Feature {col_name[i]}\nCorrelation {correlation[i]:.3f}')
        ax.legend(loc='upper left')

        ax1 = ax.twinx()
        ax1.plot(Y, label='target', c='r')
        ax1.legend(loc='upper right')

    save_path = f'./imgs/{dataset.name}.png'
    plt.savefig(save_path)
    print(f'Image is saved in {save_path}')
    # plt.show()
