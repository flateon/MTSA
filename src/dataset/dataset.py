from pathlib import Path

import numpy as np
import pandas as pd


class DatasetBase:
    def __init__(self, args):
        self.split = False
        self.read_data()
        self.split_data(args.seq_len)

    def read_data(self):
        raise NotImplementedError

    def split_data(self, seq_len):
        pass


class M4Dataset(DatasetBase):
    def __init__(self, args):
        self.train_data_path = args.train_data_path
        self.test_data_path = args.train_data_path
        self.type = 'm4'
        super().__init__(args)

    def read_data(self):
        """
        read_data function for M4 dataset(https://github.com/Mcompetitions/M4-methods/tree/master/Dataset).

        :param
            self.data_path: list, [train_data_path: str, test_data_path: str]
        :return
            self.train_data: np.ndarray, shape=(n_samples, timesteps)
            self.test_data: np.ndarray, shape=(n_samples, timesteps)
        """
        self.name = Path(self.train_data_path).stem.removesuffix('-train')
        train_data_path = self.train_data_path
        train_data = pd.read_csv(train_data_path)
        train_data.set_index('V1', inplace=True)
        self.train_data = np.array([v[~np.isnan(v)] for v in
                                    train_data.values], dtype=object)

        test_data_path = self.test_data_path
        test_data = pd.read_csv(test_data_path)
        test_data.set_index('V1', inplace=True)
        self.test_data = np.array([v[~np.isnan(v)] for v in
                                   test_data.values], dtype=object)

        self.val_data = None


class ETTDataset(DatasetBase):
    def __init__(self, args):
        self.data_path = args.data_path
        self.target = args.target
        self.type = 'ETT'
        self.frequency = args.frequency
        super(ETTDataset, self).__init__(args)

    def read_data(self):
        '''
        read_data function for ETT dataset(https://github.com/zhouhaoyi/ETDataset).

        :param
            self.data_path: str
        :return
            self.data_stamp: data timestamps
            self.data_cols: data columns(features/targets)
            self.data: np.ndarray, shape=(n_samples, timesteps, channels)
        '''
        self.name = Path(self.data_path).stem
        data = pd.read_csv(self.data_path)
        cols = list(data.columns)
        cols.remove(self.target)
        cols.remove('date')
        data = data[['date'] + cols + [self.target]]
        self.data_stamp = pd.to_datetime(data.date)
        self.data_cols = cols + [self.target]
        self.data = np.expand_dims(data[self.data_cols].values, axis=0).astype(np.float32)

    def split_data(self, seq_len):
        self.split = True
        if self.frequency == 'h':
            self.num_train = 12 * 30 * 24
            self.num_val = 4 * 30 * 24
            self.num_test = 4 * 30 * 24
        elif self.frequency == 'm':
            self.num_train = 12 * 30 * 24 * 4
            self.num_val = 4 * 30 * 24 * 4
            self.num_test = 4 * 30 * 24 * 4
        self.train_data = self.data[:, :self.num_train, :]
        self.val_data = self.data[:, self.num_train - seq_len: self.num_train + self.num_val, :]
        self.test_data = self.data[:,
                         self.num_train + self.num_val - seq_len: self.num_train + self.num_val + self.num_test, :]


class CustomDataset(DatasetBase):
    def __init__(self, args):
        self.data_path = args.data_path
        self.target = args.target
        self.type = 'Custom'
        self.name = ''
        super().__init__(args)

    def read_data(self):
        '''
        read_data function for other datasets:
            all the .csv files can be found here (https://box.nju.edu.cn/d/b33a9f73813048b8b00f/)
            Traffic (http://pems.dot.ca.gov)
            Electricity (https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)
            Exchange-Rate (https://github.com/laiguokun/multivariate-time-series-data)
            Weather (https://www.bgc-jena.mpg.de/wetter/)
            ILI(illness) (https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html)

        :param
            self.data_path: str
        :return
            self.data_stamp: data timestamps
            self.data_cols: data columns(features/targets)
            self.data: np.ndarray, shape=(n_samples, timesteps, channels), where the last channel is the target
        '''
        self.name = Path(self.data_path).stem
        data = pd.read_csv(self.data_path)
        cols = list(data.columns)
        cols.remove(self.target)
        cols.remove('date')
        data = data[['date'] + cols + [self.target]]
        self.data_stamp = pd.to_datetime(data.date)
        self.data_cols = cols + [self.target]
        self.data = np.expand_dims(data[self.data_cols].values, axis=0).astype(np.float32)

    def split_data(self, seq_len):
        self.split = True
        tot_num = self.data.shape[1]
        self.num_train = int(tot_num * 0.7)
        self.num_test = int(tot_num * 0.2)
        self.num_val = tot_num - self.num_train - self.num_test
        self.train_data = self.data[:, :self.num_train, :]
        self.val_data = self.data[:, self.num_train - seq_len: self.num_train + self.num_val, :]
        self.test_data = self.data[:, self.num_train + self.num_val - seq_len:, :]


def get_dataset(args):
    dataset_dict = {
        'M4':     M4Dataset,
        'ETT':    ETTDataset,
        'Custom': CustomDataset,
    }
    return dataset_dict[args.dataset](args)
