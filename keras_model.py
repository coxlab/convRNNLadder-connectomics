keras_dir = '/home/bill/Libraries/keras/'
import sys
sys.path.append(keras_dir)
from keras.models import *
from keras.layers.core import *
from keras.layers.convolutional import *

class ConvLSTMLadderNet(Graph):

    def __init__(self, config, build=True):
        super(ConvLSTMLadderNet, self).__init__()
        self.config = config
        self.initialize()
        if build:
            self.build()

    def initialize(self):

        # initialize hidden states
    for l in range(self.config.n_modules):
        self.add_input(name='H_l%d_t-1' % l, input_shape=(self.config.stack_sizes[l], 1024 // 2**(l+1), 1024 // 2**(l+1)))
        self.add_input(name='C_l%d_t-1' % l, input_shape=(self.config.stack_sizes[l], 1024 // 2**(l+1), 1024 // 2**(l+1)))
