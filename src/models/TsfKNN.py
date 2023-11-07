import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from src.models.base import MLForecastModel
from src.utils.distance import euclidean, manhattan, chebyshev, minkowski, cosine, DecomposeDistance, zero


class TsfKNN(MLForecastModel):
    def __init__(self, args):
        self.k = args.n_neighbors
        if args.distance == 'euclidean':
            self.distance = euclidean
        elif args.distance == 'manhattan':
            self.distance = manhattan
        elif args.distance == 'chebyshev':
            self.distance = chebyshev
        elif args.distance == 'minkowski':
            self.distance = minkowski
        elif args.distance == 'cosine':
            self.distance = cosine
        elif args.distance == 'decompose':
            self.distance = DecomposeDistance(distance=chebyshev)
        elif args.distance == 'zero':
            self.distance = zero
        self.msas = args.msas
        if args.knn == 'brute_force':
            self.knn = BruteForce(self.distance, self.k)
        elif args.knn == 'lsh':
            self.knn = LSH(args.num_bits, args.num_hashes, self.k, self.distance)
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        self.X = X[0, :, :]

    def _search(self, x, X_s, seq_len, pred_len):
        if self.msas == 'MIMO':
            indices_of_smallest_k = self.knn.query(x)
            neighbor_fore = X_s[indices_of_smallest_k, seq_len:]
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
            return x_fore
        elif self.msas == 'recursive':
            indices_of_smallest_k = self.knn.query(x)
            neighbor_fore = X_s[indices_of_smallest_k, seq_len].reshape((-1, 1))
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
            x_new = np.concatenate((x[:, 1:], x_fore), axis=1)
            if pred_len == 1:
                return x_fore
            else:
                return np.concatenate((x_fore, self._search(x_new, X_s, seq_len, pred_len - 1)), axis=1)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        fore = []
        bs, seq_len, channels = X.shape
        X_s = sliding_window_view(self.X, (seq_len + pred_len, channels)).reshape(-1, seq_len + pred_len, channels)
        for i in range(X.shape[0]):
            x = X[i, :, :]
            x_fore = self._search(x, X_s, seq_len, pred_len)
            fore.append(x_fore)
        fore = np.concatenate(fore, axis=0)
        return fore
