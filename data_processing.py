import numpy as np
from scipy.ndimage.interpolation import rotate
import pdb, traceback, sys

def create_training_set():
    P = {}
    P['version'] = 0
    P['nt_in'] = 5
    P['shape'] = (256, 256)
    P['n_examples'] = 4000
    P['orig_file'] = '/home/thouis/ForBill/train_data.h5'

    f = h5py.File(P['orig_file'], 'r')
    X_orig = f['normed_images']
    Y_orig = f['train_membrane_distance']

    X = np.zeros((P['n_examples'], P['nt_in'], 1) + P['shape'])).astype(np.float32)
    Y = np.zeros((P['n_examples'], 1) + P['shape'])).astype(np.float32)

    n = X.shape[0]
    ny = X.shape[1]
    nx = X.shape[2]
    count = 0
    while count < P['n_examples']:
        idx = np.random.randint(n - P['nt_in'] + 1)
        theta = 360*np.random.rand()
        shift_x = np.random.randint(nx - P['shape'][1])
        shift_y = np.random.randint(ny - P['shape'][0])
        test = np.ones(X_orig.shape[1:])
        test = rotate(test, theta)[shift_y:shift_y+P['shape'][0], shift_x:shift_x+P['shape'][1]]
        if np.sum(test<0.99) > 0:
            continue
        for t in range(P['nt_in']):
            X[count, t, 1] = rotate(X_orig[idx+t], theta, mode='reflect')[shift_y:shift_y+P['shape'][0], shift_x:shift_x+P['shape'][1]]
            if t==P['nt_in']-1:
                Y[count, 1] = rotate(X_orig, theta, mode='reflect')[shift_y:shift_y+P['shape'][0], shift_x:shift_x+P['shape'][1]]
        count += 1
        print count

if __name__ == "__main__":
    try:
        create_training_set()
    except:
		ty, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
