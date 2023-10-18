import unittest

import numpy as np

from src.dataset.dataset import get_dataset


class Args:
    """
    a = Args(data_path='./test')
    a.data_path => ./test
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__.update({k: v})


class TestCustomDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_conf = [('./dataset/electricity/electricity.csv', 'OT'),
                             ('./dataset/exchange_rate/exchange_rate.csv', 'OT'),
                             ('./dataset/illness/national_illness.csv', 'OT'),
                             ('./dataset/traffic/traffic.csv', 'OT'), ]
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
