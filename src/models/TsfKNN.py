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
        self.X = X

    def _search(self, x, X_s, seq_len, pred_len):
        """
        :param x: (seq_len, channels)
        :param X_s: (num_vectors, seq_len, channels)
        :param seq_len: int
        :param pred_len: int
        :return: (pred_len, channels)
        """
        if self.msas == 'MIMO':
            indices_of_smallest_k = self.knn.query(x)
            neighbor_fore = X_s[indices_of_smallest_k, seq_len:]
            x_fore = np.mean(neighbor_fore, axis=0)
            return x_fore
        elif self.msas == 'recursive':
            indices_of_smallest_k = self.knn.query(x)
            neighbor_fore = X_s[indices_of_smallest_k, seq_len:seq_len+1, :]
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=False)
            x_new = np.concatenate((x[1:], x_fore))
            if pred_len == 1:
                return x_fore
            else:
                return np.concatenate((x_fore, self._search(x_new, X_s, seq_len, pred_len - 1)), axis=0)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        fore = []
        bs, seq_len, channels = X.shape
        X_s = sliding_window_view(self.X, (seq_len + pred_len, channels), axis=(1, 2)).reshape(-1, seq_len + pred_len,
                                                                                               channels)
        self.knn.insert(X_s[:, :seq_len, :])

        for x in tqdm(X, leave=False):
            x_fore = self._search(x, X_s, seq_len, pred_len)
            fore.append(x_fore)
        fore = np.stack(fore, axis=0)
        return fore


class BruteForce:
    def __init__(self, distance, k):
        self.distance = distance
        self.k = k
        self.vectors = None

    def insert(self, vectors):
        """
        :param vectors: (num_vectors, dimensions) or (num_vectors, dimensions, channels)
        :return: None
        """
        self.vectors = vectors

    def query(self, query_vector):
        """
        :param query_vector: (dimensions, ) or (dimensions, channels)
        :return: index of top k vectors
        """
        distance = self.distance(query_vector[np.newaxis, ...], self.vectors)
        return distance.argsort()[:self.k]


class LSH:
    def __init__(self, num_bits, num_hashes, k, distance):
        """
        :param num_bits: int
        :param num_hashes: int
        :param k: int
        :param distance: function
        :return: None
        """
        self.num_bits = num_bits
        self.num_hashes = num_hashes
        self.random_vectors = None
        self.hash_tables = np.zeros((num_hashes, 2 ** num_bits), dtype=object)
        self.bit2int = np.array([2 ** i for i in range(num_bits)])
        self.mean = None
        self.vectors = None
        self.brute_force = BruteForce(distance, k)

    def generate_random_vector(self, dimensions, num):
        """
        :param dimensions: int
        :param num: number of vectors
        :return: (dimensions, num_vectors)
        """
        return np.random.normal(np.zeros(dimensions), size=(num, dimensions)).T.astype(np.float32)
        # return np.random.random((num, dimensions)).T.astype(np.float32) - 0.5

    def hash_vectors(self, vectors):
        """
        :param vectors: (num_vectors, dimensions)
        :return: index of each hash (num_hashes, num_vectors)
        """
        projection = np.dot(vectors, self.random_vectors)
        code = (projection > 0).reshape((-1, self.num_hashes, self.num_bits)).astype(np.int32)
        return code.dot(self.bit2int).T

    def insert(self, vectors):
        """
        :param vectors: (num_vectors, dimensions) or (num_vectors, dimensions, channels)
        :return: None
        """
        if len(vectors.shape) > 2:
            vectors = vectors.reshape((vectors.shape[0], -1))
        dim = vectors.shape[-1]
        self.random_vectors = self.generate_random_vector(dim, self.num_bits * self.num_hashes)
        self.vectors = vectors

        self.mean = np.mean(vectors, axis=0)
        vectors_centered = vectors - self.mean
        code = self.hash_vectors(vectors_centered)
        for i in range(self.num_hashes):
            for j in range(2 ** self.num_bits):
                self.hash_tables[i][j] = set(np.where(code[i] == j)[0])

    def query(self, query_vector, return_acc=False):
        """
        :param query_vector: (dimensions, ) or (dimensions, channels)
        :param return_acc: bool, whether to return accuracy of lsh
        :return: (nearest_neighbor, )
        """
        candidates = set()
        # Hash the query vector
        if len(query_vector.shape) > 1:
            query_vector = query_vector.reshape((-1))
        query_vector_centered = query_vector - self.mean
        code = self.hash_vectors(np.expand_dims(query_vector_centered, axis=0))[:, 0]
        for i, c in enumerate(code):
            candidates.update(self.hash_tables[i][c])

        # Brute-force search to find the exact nearest neighbors in candidates
        if len(candidates) > self.brute_force.k:
            candidates_idx = np.array(list(candidates))
            self.brute_force.insert(self.vectors[candidates_idx])
            pred = candidates_idx[self.brute_force.query(query_vector)]
        else:
            # If the number of candidates is less than k, use full brute-force search
            self.brute_force.insert(self.vectors)
            pred = self.brute_force.query(query_vector)

        if return_acc:
            self.brute_force.insert(self.vectors)
            gt = self.brute_force.query(query_vector)
            acc = np.mean(gt == pred)
            return pred, acc
        else:
            return pred
