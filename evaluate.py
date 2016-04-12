from consts import *
from config import Config
import pickle as pkl
from keras_model import *
import hickle as hkl
import traceback, sys, pdb, h5py
import matplotlib.pyplot as plt
from helpers import *


def create_predictions_from_run(run_num, data_file, out_name, overlap = 10, sig_sigma = 2):
    model, config = load_model_from_run(run_num, 'best', True)
    print 'Done loading model'
    f = h5py.File(data_file, 'r')
    X = f['normed_images']
    Y_hat = np.zeros((X.shape[0] - config.nt_in + 1,) + X.shape[1:])
    ny = config.input_shape[1]
    nx = config.input_shape[2]
    batch_X = np.zeros((1, config.nt_in, 1, ny, nx))
    sub_weights = np.ones(config.input_shape[1:])
    ramp = (np.arange(overlap, dtype=float) + 1) / overlap
    ramp = ramp.reshape((-1, 1))
    sub_weights[:overlap, :] *= ramp
    sub_weights[-overlap:, :] *= ramp[::-1]
    sub_weights.T[:overlap, :] *= ramp
    sub_weights.T[-overlap:, :] *= ramp[::-1]
    for i in range(config.nt_in - 1, X.shape[0]):
        print 'Slice '+str(i)
        hit_y_bound = False
        y_pos = 0
        total_weights = np.zeros(X.shape[1:]).astype(np.float32)
        this_slice = np.zeros(X.shape[1:]).astype(np.float32)
        while not hit_y_bound:
            y_start = y_pos * (ny - overlap)
            y_end = y_start + ny
            if y_end >= X.shape[1]:
                d = y_end - X.shape[1]
                y_start = y_start - d
                y_end = X.shape[1]
                hit_y_bound = True

            x_pos = 0
            hit_x_bound = False
            while not hit_x_bound:
                x_start = x_pos * (nx - overlap)
                x_end = x_start + nx
                if x_end >= X.shape[2]:
                    d = x_end - X.shape[2]
                    x_start = x_start - d
                    x_end = X.shape[2]
                    hit_x_bound = True

                stack = X[i - config.nt_in + 1: i+1, y_start:y_end, x_start:x_end]
                batch_X[0,:,0] = stack
                batch_Yhat = model.predict(model.format_data(batch_X))['output_t%d' % config.t_predict[0]][0,0]
                this_slice[y_start:y_end, x_start:x_end] += sub_weights * batch_Yhat
                total_weights[y_start:y_end, x_start:x_end] += sub_weights
                x_pos += 1
            y_pos += 1
        Y_hat[i - config.nt_in + 1] = this_slice / total_weights

    if config.predict_var == 'membrane':
        Y_hat[Y_hat > 1.] = 1.
        Y_hat[Y_hat < 0.] = 0.
    else:
        Y_hat = 1. - 1. / (1. + np.exp(-Y_hat / sig_sigma))
    Y_hat = Y_hat.reshape(Y_hat.shape + (1,))
    hkl.dump(Y_hat, open(get_run_dir(run_num) + out_name, 'w'))


def resave_predictions(runs):
    out_name = 'full_test_predictions.h5'
    for r in runs:
        f = get_run_dir(r) + out_name
        Y_hat = hkl.load(open(f))
        Y_hat = Y_hat.reshape(Y_hat.shape + (1,))
        hkl.dump(Y_hat, open(get_run_dir(r) + out_name, 'w'))

    data_file = '/home/thouis/ForBill/test_data.h5'
    f = h5py.File(data_file, 'r')
    M = f['membranes'][4:,:,:]
    M = M.reshape(M.shape + (1,))
    hkl.dump(M, open(nas_dir + 'bill/convRNNLadder-connectomics/data/test_membranes_t5.h5', 'w'))

def create_gt():
    data_file = '/home/thouis/ForBill/test_data.h5'
    f = h5py.File(data_file, 'r')
    M = f['membranes'][4:,:,:]
    M = M.reshape(M.shape + (1,))
    hkl.dump(M, open(nas_dir + 'bill/convRNNLadder-connectomics/data/test_membranes_t5.h5', 'w'))



if __name__ == "__main__":
    try:
        data_file = '/home/thouis/ForBill/test_data.h5'
        out_name = 'full_test_predictions.h5'
        create_predictions_from_run(47, data_file, out_name)

        #resave_predictions([35, 48, 46, 49])
    except:
		ty, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
