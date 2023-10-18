import numpy as np


class Transform:
    """
    Preprocess time series
    """

    def transform(self, data, *args, **kwargs):
        """
        :param data: raw timeseries
        :return: transformed timeseries
        """
        raise NotImplementedError

    def inverse_transform(self, data, *args, **kwargs):
        """
        :param data: raw timeseries
        :return: inverse_transformed timeseries
        """
        raise NotImplementedError


class IdentityTransform(Transform):
    def __init__(self, *args, **kwargs):
        pass

    def transform(self, data, *args, **kwargs):
        return data

    def inverse_transform(self, data, *args, **kwargs):
        return data


class LinearTransform(Transform):
    def __init__(self, weight=None, bias=None):
        self.weight = weight
        self.bias = bias

    def set_weight_bias(self, data):
        raise NotImplementedError

    def transform(self, data, target_only=False):
        """
        Param:
        data: ndarray shape=(n_samples, timestamp, channel) or (n_samples, timestamp) if target_only=True
        target_only: data is target only, no other features
        """
        if self.weight is None or self.bias is None:
            self.set_weight_bias(data)
        if target_only:
            return data * self.weight[..., -1] + self.bias[..., -1]
        else:
            return data * self.weight + self.bias

    def inverse_transform(self, data, target_only=False):
        """
        Param:
        data: ndarray shape=(n_samples, timestamp, channel)
        target_only: data is target only, no other features
        """
        if self.weight is None or self.bias is None:
            raise ValueError('Weight or bias is None')
        if target_only:
            return (data - self.bias[..., -1]) / self.weight[..., -1]
        else:
            return (data - self.bias) / self.weight


class NormalizationTransform(LinearTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def set_weight_bias(self, data):
        """
        :Param data: ndarray shape=(n_samples, timestamp, channel) or (n_samples, timestamp)
        """
        y_max = np.max(data, axis=1, keepdims=True)
        y_min = np.min(data, axis=1, keepdims=True)
        self.weight = 1 / (y_max - y_min + 1e-8)
        self.bias = -y_min * self.weight


class StandardizationTransform(LinearTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def set_weight_bias(self, data):
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        self.weight = 1 / (std + 1e-8)
        self.bias = -mean * self.weight


class MeanNormalizationTransform(LinearTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def set_weight_bias(self, data):
        mean = np.mean(data, axis=1, keepdims=True)
        y_max = np.max(data, axis=1, keepdims=True)
        y_min = np.min(data, axis=1, keepdims=True)
        self.weight = 1 / (y_max - y_min + 1e-8)
        self.bias = -mean * self.weight


class YeoJohnsonTransform(Transform):
    def __init__(self, arg, *args, **kwargs):
        self.lamda = arg.lamda

    def transform(self, data, *args, **kwargs):
        result = np.zeros_like(data)
        lamda = self.lamda
        pos_mask = data >= 0
        neg_mask = ~pos_mask

        if np.isclose(lamda, 0):
            result[pos_mask] = np.log1p(data[pos_mask])
        else:
            result[pos_mask] = (np.power(data[pos_mask] + 1, lamda) - 1) / lamda

        if np.isclose(lamda, 2):
            result[neg_mask] = -np.log1p(-data[neg_mask])
        else:
            result[neg_mask] = -(np.power(-data[neg_mask] + 1, 2 - lamda) - 1) / (2 - lamda)
        return result

    def inverse_transform(self, data, *args, **kwargs):
        result = np.zeros_like(data)
        lamda = self.lamda
        pos_mask = data >= 0
        neg_mask = ~pos_mask

        if np.isclose(lamda, 0):
            result[pos_mask] = np.exp(data[pos_mask]) - 1
        else:
            result[pos_mask] = np.power(data[pos_mask] * lamda + 1, 1 / lamda) - 1

        if np.isclose(lamda, 2):
            result[neg_mask] = 1 - np.exp(-data[neg_mask])
        else:
            result[neg_mask] = 1 - np.power(-(2 - lamda) * data[neg_mask] + 1, 1 / (2 - lamda))
        return result


class BoxCoxTransform(YeoJohnsonTransform):
    pass
