import unittest

import numpy as np
from src.utils.transforms import *


class TestTransforms(unittest.TestCase):

    def setUp(self):
        # element: [-100, 100]
        self.data = [np.random.random((1, 1000, 10)) * 200 - 100,
                     np.random.random((10, 1000, 10)) * 200 - 100,
                     np.random.random((10, 1000)) * 200 - 100]
        self.lamda = [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]

    def test_identity_transform(self):
        for data in self.data:
            identity_transform = IdentityTransform()

            transformed_data = identity_transform.transform(data)
            self.assertTrue(np.allclose(transformed_data, data))

            inv_transformed_data = identity_transform.inverse_transform(data)
            self.assertTrue(np.allclose(inv_transformed_data, data))

    def test_normalization_transform(self):
        for data in self.data:
            normalization_transform = NormalizationTransform()

            transformed_data = normalization_transform.transform(data)
            y_max = np.max(data, axis=1, keepdims=True)
            y_min = np.min(data, axis=1, keepdims=True)
            expected_transformed_data = (data - y_min) / (y_max - y_min + 1e-8)
            self.assertTrue(np.allclose(transformed_data, expected_transformed_data))

            inv_transformed_data = normalization_transform.inverse_transform(transformed_data)
            self.assertTrue(np.allclose(inv_transformed_data, data))

    def test_standardization_transform(self):
        for data in self.data:
            standardization_transform = StandardizationTransform()

            transformed_data = standardization_transform.transform(data)
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            expected_transformed_data = (data - mean) / (std + 1e-8)
            self.assertTrue(np.allclose(transformed_data, expected_transformed_data))

            inv_transformed_data = standardization_transform.inverse_transform(transformed_data)
            self.assertTrue(np.allclose(inv_transformed_data, data))

    def test_mean_normalization_transform(self):
        for data in self.data:
            mean_normalization_transform = MeanNormalizationTransform()

            transformed_data = mean_normalization_transform.transform(data)
            mean = np.mean(data, axis=1, keepdims=True)
            y_max = np.max(data, axis=1, keepdims=True)
            y_min = np.min(data, axis=1, keepdims=True)
            expected_transformed_data = (data - mean) / (y_max - y_min + 1e-8)
            self.assertTrue(np.allclose(transformed_data, expected_transformed_data))

            inv_transformed_data = mean_normalization_transform.inverse_transform(transformed_data)
            self.assertTrue(np.allclose(inv_transformed_data, data))

    def test_box_cox_transform(self):
        for data in self.data:
            for lamda in self.lamda:
                box_cox_transform = BoxCoxTransform(lamda)

                transformed_data = box_cox_transform.transform(data)
                inverse_transformed_data = box_cox_transform.inverse_transform(transformed_data)

                self.assertTrue(np.allclose(data, inverse_transformed_data))
