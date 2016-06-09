import logging
import h5py, pdb
import numpy as np
import hickle as hkl
import pdb

class ConnectomicsFromRay:
    def __init__(self, config):
        self.config = config
        self.load_data()

    def val(self):
        return (self.val_x, self.val_y)

    def train(self):
        return (self.train_x, self.train_y)

    def batch_iterator(self, epoch):
        idx = np.random.permutation(self.train_x.shape[0])[:self.config.epoch_size]
        return (self.train_x[idx], self.train_y[idx])

    def load_data(self):
        f = h5py.File(self.config.data_dir+'train_data.h5', 'r')
        X = f['normed_images']
        Y = f['train_membrane_distance']
        self.train_x = self.split_x(X[:-self.config.n_val])
        self.train_y = self.split_y(Y[:-self.config.n_val])
        self.val_x = self.split_x(X[-self.config.n_val:])
        self.val_y = self.split_y(Y[-self.config.n_val:])

    def split_x(self, raw_data):
        raw_data = raw_data[:,:self.config.input_shape[1],:self.config.input_shape[2]]
        data =  np.zeros((raw_data.shape[0] - self.config.nt_in + 1, self.config.nt_in, self.config.input_shape[0], self.config.input_shape[1], self.config.input_shape[2])).astype(np.float32)
        for t in range(self.config.nt_in):
            data[:, t, 0] = raw_data[t:(t + raw_data.shape[0] - self.config.nt_in + 1)]

        return data

    def split_y(self, raw_data):
        raw_data = raw_data[:,:self.config.input_shape[1],:self.config.input_shape[2]]
        data =  np.zeros((raw_data.shape[0] - self.config.nt_in + 1, len(self.config.t_predict), self.config.input_shape[0], self.config.input_shape[1], self.config.input_shape[2])).astype(np.float32)
        for i,t in enumerate(self.config.t_predict):
            data[:, i, 0] = raw_data[t:(t + raw_data.shape[0] - self.config.nt_in + 1)]

        return data

class PreprocessedConnectomics:
    def __init__(self, config):
        self.config = config
        self.output_map = {'distance': 'Y', 'membrane': 'M'}
        self.load_data()

    def val(self):
        return (self.val_x, self.val_y)

    def train(self):
        return (self.train_x, self.train_y)

    def batch_iterator(self, epoch):
        idx = np.random.permutation(self.train_x.shape[0])[:self.config.epoch_size]
        return (self.train_x[idx], self.train_y[idx])

    def load_data(self):
        self.train_x = hkl.load(open(self.config.train_data_dir + 'X.hkl'))[:,-self.config.nt_in:]
        self.val_x = hkl.load(open(self.config.val_data_dir + 'X.hkl'))[:40,-self.config.nt_in:]
        self.test_x = hkl.load(open(self.config.test_data_dir + 'X.hkl'))[40:80,-self.config.nt_in:]

        tag = self.output_map[self.config.predict_var]
        self.train_y = self.process(hkl.load(open(self.config.train_data_dir + tag + '.hkl')), tag)
        self.val_y = self.process(hkl.load(open(self.config.val_data_dir + tag + '.hkl'))[:40], tag)
        self.test_y = self.process(hkl.load(open(self.config.test_data_dir + tag + '.hkl'))[40:80], tag)

        # to be compatible with other stuff
        self.train_y = self.train_y.reshape( (self.train_y.shape[0], 1) + self.train_y.shape[1:])
        self.val_y = self.val_y.reshape( (self.val_y.shape[0], 1) + self.val_y.shape[1:])
        self.test_y = self.test_y.reshape( (self.test_y.shape[0], 1) + self.test_y.shape[1:])

    def process(self, data, tag):
        if tag == 'M':
            data = data[:].astype(np.float32) / 255.
        return data
