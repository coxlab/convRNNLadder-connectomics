import numpy as np
from scipy.ndimage.interpolation import rotate
import pdb, traceback, sys, os
import h5py
import hickle as hkl
import pickle as pkl

def create_dataset():
    P = {}
    P['version'] = 1
    P['nt_in'] = 5
    P['shape'] = (256, 256)
    P['n_examples'] = 10000
    P['orig_file'] = '/n/home06/matthewli/testing/AC4_images.hkl'

    f = h5py.File(P['orig_file'], 'r')
    X_orig = f['normed_images']
   # Y_orig = f['train_membrane_distance']
    M_orig = f['membranes']

    X = np.zeros((P['n_examples'], P['nt_in'], 1) + P['shape']).astype(np.float32)
   # Y = np.zeros((P['n_examples'], 1) + P['shape']).astype(np.float32)
    M = np.zeros((P['n_examples'], 1) + P['shape']).astype(np.float32)

    n = X_orig.shape[0]
    ny = X_orig.shape[1]
    nx = X_orig.shape[2]
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
        if np.random.rand() < 0.5:
            flip = True
        else:
            flip = False
        for t in range(P['nt_in']):
            X_base = X_orig[idx+t]
            if flip:
                X_base = np.fliplr(X_base)
            X[count, t, 0] = rotate(X_base, theta, mode='reflect')[shift_y:shift_y+P['shape'][0], shift_x:shift_x+P['shape'][1]]
            if t==P['nt_in']-1:
              #  Y_base = Y_orig[idx+t]
               # if flip:
                #    Y_base = np.fliplr(Y_base)
              #  Y[count, 0] = rotate(Y_base, theta, mode='reflect')[shift_y:shift_y+P['shape'][0], shift_x:shift_x+P['shape'][1]]
                M_base = M_orig[idx+t]
                if flip:
                    M_base = np.fliplr(M_base)
                M_rot = rotate(M_base, theta, mode='reflect')
                #M_rot[M_rot < 127] = 0.
                #M_rot[M_rot >= 127] = 1.
                M[count, 0] = M_rot[shift_y:shift_y+P['shape'][0], shift_x:shift_x+P['shape'][1]]
        count += 1
        print count

    out_dir = '/n/coxfs01/emily/data/version_'+str(P['version'])+'/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    hkl.dump(X, open(out_dir + 'X.hkl','w'))
  #  hkl.dump(Y, open(out_dir + 'Y.hkl','w'))
    hkl.dump(M, open(out_dir + 'M.hkl','w'))
    pkl.dump(P, open(out_dir + 'P.pkl','w'))

if __name__ == "__main__":
    try:
        create_dataset()
    except:
        ty, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
