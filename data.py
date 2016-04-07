import logging
import h5py, pdb
import numpy as np

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
