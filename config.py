import sys
sys.path.append('../')
from mypyutils.basic import get_next_num
from consts import *
from callback import *
from keras_model import *
from data import *

class Config(object):
    def __init__(self, param_overrides=None):
        run_num = get_next_num(base_run_dir+'run_')
        self.name = 'run_%d' % run_num
        self.tag = 'default'
        self.data_folder = os.path.join(os.getenv('HOME'), 'Data/FaceGen_Rotations/clipset4/')
        self.save_dir = os.path.join(base_run_dir, self.name + '/')

        self.t_predict = [4]
        self.loss_weights = [1.0]
        self.loss = 'mae'
        self.optimizer = 'adam'
        self.nt_in = 5
        self.n_modules = 2
        self.stack_sizes = {-1: 1, 0: 16, 1: 32}

        self.max_epochs = 50
        self.batch_size = 6
        self.epoch_size = 400
        self.use_batch_iterator = True
        self.patience = 50

        self.input_shape = (1, 256, 256)
        self.n_val = 20
        self.n_plot = 5

        self.epoch_callback = None
        self.epoch_callback_params = None

        self.save_last_weights = False
        self.save_best_weights = False
        self.save_predictions = True

        self.callback = basic_callback
        self.model = ConvLSTMLadderNet
        self.data = ConnectomicsFromRay
        self.data_dir = '/home/thouis/ForBill/'

    def __str__(self):
        return self.__dict__.__str__()

    def write_to_file(self):
        f = open(os.path.join(self.save_dir, 'config.txt'), 'w')
        for key in self.__dict__:
            f.write("%s\t%s\n" % (key, self.__dict__[key]))
        f.close()
