import numpy as np

from src.models.DLinear import DLinear as DLinear
from src.models.base import MLForecastModel
from sklearn.decomposition import PCA


class SPIRIT(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        if args.pca_components in ('mle', None):
            n_components = args.pca_components
        else:
            n_components = float(args.pca_components)
            if n_components >= 1.0:
                n_components = int(n_components)
        self.pca = PCA(n_components=n_components)
        self.model = DLinear(args)
        self.n_components = None

    def _fit(self, X: np.ndarray, args) -> None:
        n_samples, seq_len, n_channels = X.shape
        X = X.reshape(-1, n_channels)
        X_pca = self.pca.fit_transform(X).reshape(n_samples, seq_len, -1)
        self.n_components = self.pca.n_components_
        self.model.fit(X_pca, args)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        n_samples, seq_len, n_channels = X.shape
        X = X.reshape(-1, n_channels)
        X_pca = self.pca.transform(X).reshape(n_samples, seq_len, -1)
        pred = self.model.forecast(X_pca, pred_len)
        return self.pca.inverse_transform(pred)
