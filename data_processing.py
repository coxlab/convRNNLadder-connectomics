


def create_training_set():
    P = {}
    P['version'] = 0
    P['nt_in'] = 5
    P['shape'] = (256, 256)
    P['n_examples'] =
    P['orig_file'] = '/home/thouis/ForBill/train_data.h5'

    f = h5py.File(self.config.data_dir+'train_data.h5', 'r')
    X = f['normed_images']
    Y = f['train_membrane_distance']
