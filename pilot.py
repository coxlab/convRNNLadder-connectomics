keras_dir = '/home/bill/Libraries/keras/'
import sys
sys.path.append(keras_dir)
from keras.models import *
from keras.layers.core import *
from keras.layers.convolutional import *

n_modules = 2
nt_in = 6
t_predict = [4, 5]

model = Graph()



for t in range(nt_in):
    model.add_input(name='input_t%d' % t, input_shape=(1, 1023, 1024))
    for l in n_modules:
        model.add_node(name='conv0_l%d_t%d' % (l, t), input='input_t%d' % t)
