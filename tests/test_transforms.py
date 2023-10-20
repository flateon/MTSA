import unittest

import numpy as np
from src.utils.transforms import *
from argparse import Namespace as Args


class TestTransforms(unittest.TestCase):

    def setUp(self):
        # element: [-100, 100]
        self.data = [
            np.random.random((1, 1000, 10)) * 20 - 10,
            np.random.random((10, 100, 10)) * 20 - 30,
            np.random.random((1, 1000, 10)) * 0.2 - 0.1,
            np.random.random((1, 1000, 10)) * 2e-9 - 1e-9,
            np.random.random((1, 1000, 10)) * 2e9 - 1e9,
        ]
        self.lamda = [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]

    def test_transform(self):
        t = Transform()
        self.assertRaises(NotImplementedError, t.transform, None)
        self.assertRaises(NotImplementedError, t.inverse_transform, None)

    def test_identity_transform(self):
        for data in self.data:
            identity_transform = IdentityTransform()

            transformed_data = identity_transform.transform(data)
            self.assertTrue(np.allclose(transformed_data, data))

            inv_transformed_data = identity_transform.inverse_transform(data)
            self.assertTrue(np.allclose(inv_transformed_data, data))

    def test_linear_transform(self):
        t = LinearTransform()
        self.assertRaises(NotImplementedError, t.set_weight_bias, None)
        self.assertRaises(ValueError, t.inverse_transform, None)

        for data in self.data:
            weight_like = np.random.random((data.shape[0], 1, data.shape[2]))
            linear_transform = LinearTransform(weight=weight_like, bias=weight_like)

            transformed_data = linear_transform.transform(data)
            expected_transformed_data = data * linear_transform.weight + linear_transform.bias
            self.assertTrue(np.allclose(transformed_data, expected_transformed_data))

            inv_transformed_data = linear_transform.inverse_transform(transformed_data)
            self.assertTrue(np.allclose(inv_transformed_data, data))

            target = data[..., -1]
            transformed_target = linear_transform.transform(target, target_only=True)
            self.assertTrue(np.allclose(transformed_target, transformed_data[..., -1]))

            inv_transformed_target = linear_transform.inverse_transform(transformed_target, target_only=True)
            self.assertTrue(np.allclose(inv_transformed_target, target))

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

            target = data[..., -1]
            transformed_target = normalization_transform.transform(target, target_only=True)
            self.assertTrue(np.allclose(transformed_target, transformed_data[..., -1]))

            inv_transformed_target = normalization_transform.inverse_transform(transformed_target, target_only=True)
            self.assertTrue(np.allclose(inv_transformed_target, target))

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

            target = data[..., -1]
            transformed_target = standardization_transform.transform(target, target_only=True)
            self.assertTrue(np.allclose(transformed_target, transformed_data[..., -1]))

            inv_transformed_target = standardization_transform.inverse_transform(transformed_target, target_only=True)
            self.assertTrue(np.allclose(inv_transformed_target, target))

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

            target = data[..., -1]
            transformed_target = mean_normalization_transform.transform(target, target_only=True)
            self.assertTrue(np.allclose(transformed_target, transformed_data[..., -1]))

            inv_transformed_target = mean_normalization_transform.inverse_transform(transformed_target,
                                                                                    target_only=True)
            self.assertTrue(np.allclose(inv_transformed_target, target))

    def test_box_cox_transform(self):
        for data in self.data:
            for lamda in self.lamda:
                box_cox_transform = BoxCoxTransform(Args(lamda=lamda))

                transformed_data = box_cox_transform.transform(data)
                inverse_transformed_data = box_cox_transform.inverse_transform(transformed_data)

                self.assertTrue(np.allclose(data, inverse_transformed_data))

                target = data[..., -1]
                transformed_target = box_cox_transform.transform(target, target_only=True)
                self.assertTrue(np.allclose(transformed_target, transformed_data[..., -1]))

                inv_transformed_target = box_cox_transform.inverse_transform(transformed_target, target_only=True)
                self.assertTrue(np.allclose(inv_transformed_target, target))
