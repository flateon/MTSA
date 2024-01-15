import unittest

import numpy as np

from src.dataset.dataset import get_dataset
from src.dataset.data_visualizer import data_visualize, pearson_correlation
from argparse import Namespace as Args


class TestPearsonCorrelation(unittest.TestCase):
    def test_correlation_coefficient(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([[10], [20], [30]])
        correlations = pearson_correlation(X, y)

        # You should assert the expected values for the correlation coefficients
        # based on the input X and y
        self.assertEqual(correlations.shape, (3,))  # Check the shape of the result
        self.assertAlmostEqual(correlations[0], 1.0, places=2)  # Example correlation coefficient
        # Add more assertions for other correlation coefficients as needed

    def test_correlation_coefficient_with_zeros(self):
        # Test with a case where X has zero variance (all elements are the same)
        X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]).T
        y = np.array([[10], [20], [30]])
        correlations = pearson_correlation(X, y)

        # Check that the correlation coefficient is 0 in this case
        self.assertEqual(correlations.shape, (3,))
        self.assertAlmostEqual(correlations[0], 0.0, places=2)


class TestVisualizer(unittest.TestCase):
    def test_multi_channel(self):
        self.dataset_conf = [
            {'data_path': './dataset/electricity/electricity.csv', 'dataset': 'Custom'},
            {'data_path': './dataset/exchange_rate/exchange_rate.csv', 'dataset': 'Custom'},
            {'data_path': './dataset/illness/national_illness.csv', 'dataset': 'Custom'},
            {'data_path': './dataset/traffic/traffic.csv', 'dataset': 'Custom'},
            {'data_path': './dataset/weather/weather.csv', 'dataset': 'Custom'},
            {'data_path': './dataset/ETT/ETTh1.csv', 'dataset': 'ETT', 'frequency': 'h'},
            {'data_path': './dataset/ETT/ETTh2.csv', 'dataset': 'ETT', 'frequency': 'h'},
            {'data_path': './dataset/ETT/ETTm1.csv', 'dataset': 'ETT', 'frequency': 'm'},
            {'data_path': './dataset/ETT/ETTm2.csv', 'dataset': 'ETT', 'frequency': 'm'},
        ]
        args = Args(ratio_train=0.7, ratio_val=0.1, ratio_test=0.2, dataset='', data_path='', target='OT', seq_len=96, pred_len=96)
        for kwargs in self.dataset_conf:
            for k, v in kwargs.items():
                setattr(args, k, v)
            dataset_list = get_dataset(args)
            if not isinstance(dataset_list, list):
                dataset_list = [dataset_list]
            for dataset in dataset_list:
                data_visualize(dataset, 200)
        self.assertEqual('y', input('Is all dataset plot correctly? (y/n)'))

    def test_single_channel(self):
        args = Args(ratio_train=0.7, ratio_val=0.1, ratio_test=0.2, dataset='M4', seq_len=96, pred_len=96,
                    train_data_path='./dataset/m4/Hourly-train.csv', test_data_path='./dataset/m4/Hourly-test.csv')
        dataset_list = get_dataset(args)
        if not isinstance(dataset_list, list):
            dataset_list = [dataset_list]
        for d in dataset_list:
            data_visualize(d, 30)
        self.assertEqual('y', input('Is the dataset plot correctly? (y/n)'))
