import unittest

import numpy as np

from src.dataset.dataset import get_dataset, DatasetBase
from argparse import Namespace as Args


class TestCustomDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_conf = [
            {'data_path': './dataset/electricity/electricity.csv', },
            {'data_path': './dataset/exchange_rate/exchange_rate.csv', },
            {'data_path': './dataset/illness/national_illness.csv', },
            {'data_path': './dataset/traffic/traffic.csv', },
            {'data_path': './dataset/weather/weather.csv', },
        ]
        self.ratio_conf = [(0.7, 0.1, 0.2),
                           (0.7, 0, 0.3),
                           (0.999999, 0, 0.000001)]
        self.args = Args(ratio_train=0.7, ratio_val=0.1, ratio_test=0.2, dataset='Custom', data_path='', target='OT',
                         seq_len=96, pred_len=96)

    def test_read_data(self):
        for kwargs in self.dataset_conf:
            for k, v in kwargs.items():
                setattr(self.args, k, v)
            dataset = get_dataset(self.args)

            # self.data_cols: data columns(features / targets)
            self.assertIsInstance(dataset.data_cols, list)
            self.assertEqual(self.args.target, dataset.data_cols[-1])
            self.assertNotIn('date', dataset.data_cols)
            # self.data: np.ndarray, shape = (n_samples, timesteps, channels)
            self.assertIsInstance(dataset.data, np.ndarray)
            self.assertEqual(3, len(dataset.data.shape))
            self.assertEqual(1, dataset.data.shape[0])
            self.assertEqual(len(dataset.data_stamp), dataset.data.shape[1])
            self.assertEqual(len(dataset.data_cols), dataset.data.shape[2])

    # def test_split_data(self):
    #     for kwargs in self.dataset_conf:
    #         for k, v in kwargs.items():
    #             setattr(self.args, k, v)
    #         for train, val, test in self.ratio_conf:
    #             self.args.ratio_train = train
    #             self.args.ratio_val = val
    #             self.args.ratio_test = test
    #             dataset = get_dataset(self.args)
    #
    #             num_all = dataset.data.shape[1]
    #             num_train = dataset.train_data.shape[1]
    #             num_test = dataset.test_data.shape[1]
    #
    #             self.assertTrue(dataset.split)
    #             self.assertGreater(num_train, 0)
    #             self.assertGreater(num_test, 0)
    #             self.assertAlmostEqual(num_train, num_all * self.args.ratio_train, delta=2)
    #             self.assertAlmostEqual(num_test, num_all * self.args.ratio_test, delta=2)
    #
    #             if self.args.ratio_val == 0:
    #                 self.assertIsNone(dataset.val_data)
    #                 self.assertEqual(num_all, num_train + num_test)
    #             else:
    #                 num_val = dataset.val_data.shape[1]
    #                 self.assertAlmostEqual(num_val, num_all * self.args.ratio_val, delta=2)
    #                 self.assertEqual(num_all, num_train + num_val + num_test)


class TestETTDataset(TestCustomDataset):
    def setUp(self):
        self.dataset_conf = [
            {'data_path': './dataset/ETT/ETTh1.csv', 'frequency': 'h'},
            {'data_path': './dataset/ETT/ETTh2.csv', 'frequency': 'h'},
            {'data_path': './dataset/ETT/ETTm1.csv', 'frequency': 'm'},
            {'data_path': './dataset/ETT/ETTm2.csv', 'frequency': 'm'},
        ]
        self.args = Args(ratio_train=0.7, ratio_val=0.1, ratio_test=0.2, dataset='ETT', data_path='', target='OT',
                         seq_len=96, pred_len=96)

    # def test_split_data(self):
    #     for kwargs in self.dataset_conf:
    #         for k, v in kwargs.items():
    #             setattr(self.args, k, v)
    #
    #         dataset = get_dataset(self.args)
    #
    #         num_all = dataset.data.shape[1]
    #         num_train = dataset.train_data.shape[1]
    #         num_val = dataset.val_data.shape[1]
    #         num_test = dataset.test_data.shape[1]
    #
    #         if self.args.frequency == 'h':
    #             num_train_t = 18 * 30 * 24
    #             num_val_t = 5 * 30 * 24
    #         elif self.args.frequency == 'm':
    #             num_train_t = 18 * 30 * 24 * 4
    #             num_val_t = 5 * 30 * 24 * 4
    #
    #         self.assertTrue(dataset.split)
    #         self.assertGreater(num_train, 0)
    #         self.assertGreater(num_test, 0)
    #         self.assertAlmostEqual(num_train, num_train_t, delta=2)
    #         self.assertAlmostEqual(num_val, num_val_t, delta=2)
    #         self.assertEqual(num_all, num_train + num_val + num_test)


class TestM4Dataset(unittest.TestCase):
    def setUp(self):
        self.dataset_conf = [
            {'train_data_path': './dataset/m4/Yearly-train.csv', 'test_data_path': './dataset/m4/Yearly-test.csv'},
            {'train_data_path': './dataset/m4/Quarterly-train.csv',
             'test_data_path':  './dataset/m4/Quarterly-test.csv'},
            {'train_data_path': './dataset/m4/Monthly-train.csv', 'test_data_path': './dataset/m4/Monthly-test.csv'},
            {'train_data_path': './dataset/m4/Weekly-train.csv', 'test_data_path': './dataset/m4/Weekly-test.csv'},
            {'train_data_path': './dataset/m4/Daily-train.csv', 'test_data_path': './dataset/m4/Daily-test.csv'},
            {'train_data_path': './dataset/m4/Hourly-train.csv', 'test_data_path': './dataset/m4/Hourly-test.csv'},
        ]
        self.args = Args(ratio_train=0.7, ratio_val=0.1, ratio_test=0.2, dataset='M4', train_data_path='',
                         test_data_path='', seq_len=96, pred_len=96)

    def test_read_data(self):
        for kwargs in self.dataset_conf:
            for k, v in kwargs.items():
                setattr(self.args, k, v)
            dataset = get_dataset(self.args)

            # self.train_data: np.ndarray, shape=(n_samples, ) object: np.ndarray, shape=(timestamp, )
            # self.test_data: np.ndarray, shape=(n_samples, )
            self.assertIsInstance(dataset.train_data, np.ndarray)
            self.assertIsInstance(dataset.test_data, np.ndarray)
            self.assertIsInstance(dataset.train_data[0], np.ndarray)
            self.assertIsNone(dataset.val_data)
            self.assertEqual(1, len(dataset.train_data.shape))
            self.assertEqual(1, len(dataset.train_data[0].shape))
            self.assertEqual(1, len(dataset.test_data.shape))
            self.assertEqual(1, len(dataset.test_data[0].shape))
            self.assertTrue(isinstance(dataset.test_data[0], np.ndarray))


class TestDatasetBase(unittest.TestCase):
    def test_read_data(self):
        args = Args(ratio_train=0.7, ratio_val=0.1, ratio_test=0.2)
        self.assertRaises(NotImplementedError, DatasetBase, args)
