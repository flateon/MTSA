import unittest

import numpy as np

from src.dataset.dataset import get_dataset, DatasetBase
from argparse import Namespace as Args


class TestCustomDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_conf = [('./dataset/electricity/electricity.csv', 'OT'),
                             ('./dataset/exchange_rate/exchange_rate.csv', 'OT'),
                             ('./dataset/illness/national_illness.csv', 'OT'),
                             ('./dataset/traffic/traffic.csv', 'OT'),
                             ('./dataset/weather/weather.csv', 'OT'), ]
        self.ratio_conf = [(0.7, 0.1, 0.2),
                           (0.7, 0, 0.3),
                           (0.999999, 0, 0.000001)]
        self.args = Args(ratio_train=0.7, ratio_val=0.1, ratio_test=0.2, dataset='Custom', data_path='', target='')

    def test_read_data(self):
        for path, ot in self.dataset_conf:
            self.args.data_path = path
            self.args.target = ot
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

    def test_split_data(self):
        for path, ot in self.dataset_conf:
            self.args.data_path = path
            self.args.target = ot
            for train, val, test in self.ratio_conf:
                self.args.ratio_train = train
                self.args.ratio_val = val
                self.args.ratio_test = test
                dataset = get_dataset(self.args)

                num_all = dataset.data.shape[1]
                num_train = dataset.train_data.shape[1]
                num_test = dataset.test_data.shape[1]

                self.assertTrue(dataset.split)
                self.assertGreater(num_train, 0)
                self.assertGreater(num_test, 0)
                self.assertAlmostEqual(num_train, num_all * self.args.ratio_train, delta=2)
                self.assertAlmostEqual(num_test, num_all * self.args.ratio_test, delta=2)

                if self.args.ratio_val == 0:
                    self.assertIsNone(dataset.val_data)
                    self.assertEqual(num_all, num_train + num_test)
                else:
                    num_val = dataset.val_data.shape[1]
                    self.assertAlmostEqual(num_val, num_all * self.args.ratio_val, delta=2)
                    self.assertEqual(num_all, num_train + num_val + num_test)


class TestETTDataset(TestCustomDataset):
    def setUp(self):
        self.dataset_conf = [('./dataset/ETT/ETTh1.csv', 'OT'),
                             ('./dataset/ETT/ETTh2.csv', 'OT'),
                             ('./dataset/ETT/ETTm1.csv', 'OT'),
                             ('./dataset/ETT/ETTm2.csv', 'OT'), ]
        self.ratio_conf = [(0.7, 0.1, 0.2),
                           (0.7, 0, 0.3),
                           (0.999999, 0, 0.000001)]
        self.args = Args(ratio_train=0.7, ratio_val=0.1, ratio_test=0.2, dataset='ETT', data_path='', target='')


class TestM4Dataset(unittest.TestCase):
    def setUp(self):
        self.dataset_conf = [
            ('./dataset/m4/Yearly-train.csv', './dataset/m4/Yearly-test.csv'),
            ('./dataset/m4/Quarterly-train.csv', './dataset/m4/Quarterly-test.csv'),
            ('./dataset/m4/Monthly-train.csv', './dataset/m4/Monthly-test.csv'),
            ('./dataset/m4/Weekly-train.csv', './dataset/m4/Weekly-test.csv'),
            ('./dataset/m4/Daily-train.csv', './dataset/m4/Daily-test.csv'),
            ('./dataset/m4/Hourly-train.csv', './dataset/m4/Hourly-test.csv'),
        ]
        self.args = Args(ratio_train=0.7, ratio_val=0.1, ratio_test=0.2, dataset='M4', train_data_path='',
                         test_data_path='')

    def test_read_data(self):
        for train_path, test_path in self.dataset_conf:
            self.args.train_data_path = train_path
            self.args.test_data_path = test_path
            dataset = get_dataset(self.args)

            # self.train_data: np.ndarray, shape=(n_samples, ) object: np.ndarray, shape=(timestamp, )
            # self.test_data: np.ndarray, shape=(n_samples, )
            self.assertIsInstance(dataset.train_data, np.ndarray)
            self.assertIsInstance(dataset.test_data, np.ndarray)
            self.assertIsInstance(dataset.train_data[0], np.ndarray)
            self.assertIsNone(dataset.val_data)
            self.assertEqual(1, len(dataset.train_data.shape))
            self.assertEqual(1, len(dataset.train_data[0].shape))
            self.assertEqual(2, len(dataset.test_data.shape))


class TestDatasetBase(unittest.TestCase):
    def test_read_data(self):
        args = Args(ratio_train=0.7, ratio_val=0.1, ratio_test=0.2)
        self.assertRaises(NotImplementedError, DatasetBase, args)
