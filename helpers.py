from consts import *
from config import Config
import pickle as pkl
from keras_model import *
import hickle as hkl
import traceback, sys, pdb, h5py
import matplotlib.pyplot as plt


def load_model_from_run(run_num, model_str, build=False):
    run_dir = get_run_dir(run_num)
    config = pkl.load(open(run_dir + 'config.pkl'))
    model = config.model(config, build)
    model.load_weights(run_dir + 'model_weights_' + model_str + '.hdf5')

    return model, config


def get_run_dir(run_num):
    return base_run_dir + 'run_' + str(run_num) + '/'
