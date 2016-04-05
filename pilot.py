keras_dir = '/home/bill/Libraries/keras/'
import sys, traceback, pdb
sys.path.append(keras_dir)
from keras.models import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.optimizers import *
from scipy.io import savemat

import h5py

data_dir = '/home/thouis/ForBill/'

n_modules = 2
nt_in = 5
t_predict = [4]
stack_sizes = {-1: 1, 0: 16, 1: 32}
batch_size = 4
nb_epoch = 400
n_val = 20
input_size = 256

def initialize_model():

    model = Graph()

    # initialize hidden states
    for l in range(n_modules):
        model.add_input(name='H_l%d_t-1' % l, input_shape=(stack_sizes[l], input_size // 2**(l+1), input_size // 2**(l+1)))
        model.add_input(name='C_l%d_t-1' % l, input_shape=(stack_sizes[l], input_size // 2**(l+1), input_size // 2**(l+1)))

    for t in range(nt_in):
        model.add_input(name='input_t%d' % t, input_shape=(1, input_size, input_size))

        # add first conv layer
        if t == 0:
            trainable = True
            shared_layer = None
        else:
            trainable = False
            shared_layer = model.nodes['conv0pre_l-1_t0']
        layer = Convolution2D(stack_sizes[0], 5, 5, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layer)
        model.add_node(layer, name='conv0pre_l-1_t%d' % t, input='input_t%d' % t)
        model.add_node(AveragePooling2D(), name='conv0_l-1_t%d' % t, input='conv0pre_l-1_t%d' % t)

        for l in range(n_modules):
            layer_names = ['conv%d_l%d_t%d' % (i, l, t) for i in range(4)]
            if t == 0:
                trainable = True
                shared_layers = [None for _ in range(4)]
            else:
                trainable = False
                shared_layers = [model.nodes['conv%d_l%d_t0' % (i, l)] for i in range(4)]

            if l==0:
                module_input = 'conv0_l-1_t%d' % t
                module_input_upchannel = 'conv0_l-1_t%d' % t
            else:
                module_input = 'H_l%d_t%d' % (l-1, t)
                module_input_upchannel = 'Hupchannel_l%d_t%d' % (l-1, t)
                if t == 0:
                    shared_l = None
                else:
                    shared_l = model.nodes['Hupchannel_l%d_t0' % (l-1)]
                layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_l)
                model.add_node(layer, name='Hupchannel_l%d_t%d' % (l-1, t), input=module_input)


            layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[0])
            model.add_node(layer, name=layer_names[0], input=module_input)
            layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[1])
            model.add_node(layer, name=layer_names[1], input=layer_names[0])
            model.add_node(Activation('linear'), name='res0_l%d_t%d' % (l, t), inputs=[module_input_upchannel, layer_names[1]], merge_mode='sum')

            layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[2])
            model.add_node(layer, name=layer_names[2], input='res0_l%d_t%d' % (l, t))
            layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[3])
            model.add_node(layer, name=layer_names[3], input=layer_names[2])
            if l > 0:
                model.add_node(AveragePooling2D(), name='res1_l%d_t%d' % (l, t), inputs=['res0_l%d_t%d' % (l, t), layer_names[3]], merge_mode='sum')
            else:
                model.add_node(Activation('linear'), name='res1_l%d_t%d' % (l, t), inputs=['res0_l%d_t%d' % (l, t), layer_names[3]], merge_mode='sum')


            if t==0:
                shared_layers = [None for _ in range(4)]
            else:
                shared_layers = [model.nodes['I_l%d_t0' % l], model.nodes['F_l%d_t0' % l], model.nodes['O_l%d_t0' % l], model.nodes['C1_l%d_t0' % l] ]

            layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='hard_sigmoid', trainable=trainable, shared_layer=shared_layers[0])
            model.add_node(layer, name='I_l%d_t%d' % (l, t), inputs=['res1_l%d_t%d' % (l, t), 'H_l%d_t%d' % (l, t-1)], merge_mode='concat', concat_axis=-3)
            layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='hard_sigmoid', trainable=trainable, shared_layer=shared_layers[1])
            model.add_node(layer, name='F_l%d_t%d' % (l, t), inputs=['res1_l%d_t%d' % (l, t), 'H_l%d_t%d' % (l, t-1)], merge_mode='concat', concat_axis=-3)
            layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='hard_sigmoid', trainable=trainable, shared_layer=shared_layers[2])
            model.add_node(layer, name='O_l%d_t%d' % (l, t), inputs=['res1_l%d_t%d' % (l, t), 'H_l%d_t%d' % (l, t-1)], merge_mode='concat', concat_axis=-3)

            layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='tanh', trainable=trainable, shared_layer=shared_layers[3])
            model.add_node(layer, name='C1_l%d_t%d' % (l, t), inputs=['res1_l%d_t%d' % (l, t), 'H_l%d_t%d' % (l, t-1)], merge_mode='concat', concat_axis=-3)

            model.add_node(Activation('linear'), name='C2_l%d_t%d' % (l, t), inputs=['I_l%d_t%d' % (l, t), 'C1_l%d_t%d' % (l, t)], merge_mode='mul')
            model.add_node(Activation('linear'), name='C3_l%d_t%d' % (l, t), inputs=['F_l%d_t%d' % (l, t), 'C_l%d_t%d' % (l, t-1)], merge_mode='mul')
            model.add_node(Activation('linear'), name='C_l%d_t%d' % (l, t), inputs=['C3_l%d_t%d' % (l, t), 'C2_l%d_t%d' % (l, t)], merge_mode='sum')
            model.add_node(Activation('tanh'), name='Ct_l%d_t%d' % (l, t), input='C_l%d_t%d' % (l, t))

            model.add_node(Activation('linear'), name='H_l%d_t%d' % (l, t), inputs=['O_l%d_t%d' % (l, t), 'Ct_l%d_t%d' % (l, t)], merge_mode='mul')


            # layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='hard_sigmoid', trainable=trainable, shared_layer=shared_layer[0], subsample=subsample)
            # model.add_node(layer, name='z_l%d_t%d' % (l, t), inputs=, merge_mode='concat', concat_axis=-3)
            # layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='hard_sigmoid', trainable=trainable, shared_layer=shared_layer[0], subsample=subsample)
            # model.add_node(layer, name='r_l%d_t%d' % (l, t), inputs=, merge_mode='concat', concat_axis=-3)
            # layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', trainable=trainable, shared_layer=shared_layer[0], subsample=subsample)
            # model.add_node(layer, name='hu_l%d_t%d' % (l, t), inputs=, merge_mode='mul')
            # layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', trainable=trainable, shared_layer=shared_layer[0], subsample=subsample)
            # model.add_node(layer, name='hw_l%d_t%d' % (l, t), input=)
            # model.add_node(Activation('tanh'), name='htilde_l%d_t%d' % (l, t), inputs=[], merge_mode='sum')
            # layer_f = Convolution2D(stack_sizes[l], R_filt_sizes[i], R_filt_sizes[i], border_mode='same', activation='hard_sigmoid', trainable=trainable, shared_layer=shared_layer_f

        if t in t_predict:
            for l in range(n_modules-1, -1, -1):
                layer_names = ['deconv%d_l%d_t%d' % (i, l, t) for i in range(4)]
                if t == t_predict[0]:
                    trainable = True
                    shared_layers = [None for _ in range(4)]
                else:
                    trainable = False
                    shared_layers = [model.nodes['deconv%d_l%d_t%d' % (i, l, t_predict[0])] for i in range(4)]


                if l==n_modules-1:
                    module_input = 'Hup_l%d_t%d' % (l, t)
                else:
                    module_input = 'comb_l%d_t%d' % (l, t)


                model.add_node(UpSampling2D(), name='Hup_l%d_t%d' % (l, t), input='H_l%d_t%d' % (l, t))
                if l<n_modules-1:
                   model.add_node(UpSampling2D(), name='deres1up_l%d_t%d' % (l+1, t), input='deres1_l%d_t%d' % (l+1, t))

                if l<n_modules-1:
                    #model.add_node(Activation('linear'), name='prod_l%d_t%d' % (l, t), inputs=['Hup_l%d_t%d' % (l, t), 'deres1_l%d_t%d' % (l+1, t)], merge_mode='mul')
                    if t == t_predict[0]:
                        shared_l = None
                        tr = True
                    else:
                        shared_l = model.nodes['comb_l%d_t%d' % (l, t_predict[0])]
                        tr = False
                    layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=tr, shared_layer=shared_l)
                    model.add_node(layer, name='comb_l%d_t%d' % (l, t), inputs=['Hup_l%d_t%d' % (l, t), 'deres1up_l%d_t%d' % (l+1, t)], merge_mode='concat', concat_axis=-3)


                layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[0])
                model.add_node(layer, name=layer_names[0], input=module_input)
                layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[1])
                model.add_node(layer, name=layer_names[1], input=layer_names[0])
                model.add_node(Activation('linear'), name='deres0_l%d_t%d' % (l, t), inputs=[module_input, layer_names[1]], merge_mode='sum')

                layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[2])
                model.add_node(layer, name=layer_names[2], input='deres0_l%d_t%d' % (l, t))
                layer = Convolution2D(stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[3])
                model.add_node(layer, name=layer_names[3], input=layer_names[2])

                model.add_node(Activation('linear'), name='deres1_l%d_t%d' % (l, t), inputs=['deres0_l%d_t%d' % (l, t), layer_names[3]], merge_mode='sum')

                if l==0:
                    model.add_node(Convolution2D(1, 1, 1), name='output_t%d' % t, input='deres1_l%d_t%d' % (l, t), create_output=True)

    return model


def train():

    model = initialize_model()
    loss = {}
    for t in t_predict:
        loss['output_t%d' % t] = 'mae'

    print 'Compiling...'
    #optimizer = RMSprop(lr=0.0002)
    model.compile(loss=loss, optimizer='adam')

    f = h5py.File(data_dir+'train_data.h5', 'r')
    X = f['normed_images']
    y = f['train_membrane_distance']
    X = X[:-n_val][:,:input_size,:input_size]
    y = y[:-n_val][:,:input_size,:input_size]
    n = X.shape[0] - nt_in

    data = {}
    for t in range(nt_in):
        data['input_t%d' % t] = np.zeros((n - nt_in + 1, 1, input_size, input_size)).astype(np.float32)
        data['input_t%d' % t][:, 0, :, :] = X[t:t+n-nt_in+1]
        #data['input_t%d' % t][:, 0, -1] = data['input_t%d' % t][:, 0, -2]

    for l in range(n_modules):
        data['H_l%d_t-1' % l] = np.zeros((n - nt_in + 1, stack_sizes[l], input_size // 2**(l+1), input_size // 2**(l+1)))
        data['C_l%d_t-1' % l] = np.zeros((n - nt_in + 1, stack_sizes[l], input_size // 2**(l+1), input_size // 2**(l+1)))

    for t in t_predict:
        data['output_t%d' % t] = np.zeros((n - nt_in + 1, 1, input_size, input_size)).astype(np.float32)
        data['output_t%d' % t][:, 0, :, :] = y[t:t+n-nt_in+1]
        #data['output_t%d' % t][:, 0, -1] = data['output_t%d' % t][:, 0, -2]

    model.fit(data, batch_size=batch_size, nb_epoch=nb_epoch)

    return model


def evaluate(model):
    f = h5py.File(data_dir+'train_data.h5', 'r')
    X = f['normed_images']
    y = f['train_membrane_distance']
    X = X[-n_val:][:,:input_size,:input_size]
    y = y[-n_val:][:,:input_size,:input_size]
    n = X.shape[0] - nt_in
    n = X.shape[0] - nt_in

    data = {}
    for t in range(nt_in):
        data['input_t%d' % t] = np.zeros((n - nt_in + 1, 1, input_size, input_size)).astype(np.float32)
        data['input_t%d' % t][:, 0, :, :] = X[t:t+n-nt_in+1]
        #data['input_t%d' % t][:, 0, -1] = data['input_t%d' % t][:, 0, -2]

    for l in range(n_modules):
        data['H_l%d_t-1' % l] = np.zeros((n - nt_in + 1, stack_sizes[l], input_size // 2**(l+1), input_size // 2**(l+1)))
        data['C_l%d_t-1' % l] = np.zeros((n - nt_in + 1, stack_sizes[l], input_size // 2**(l+1), input_size // 2**(l+1)))

    for t in t_predict:
        data['output_t%d' % t] = np.zeros((n - nt_in + 1, 1, input_size, input_size)).astype(np.float32)
        data['output_t%d' % t][:, 0, :, :] = y[t:t+n-nt_in+1]
        #data['output_t%d' % t][:, 0, -1] = data['output_t%d' % t][:, 0, -2]

    vals = model.evaluate(data, batch_size=batch_size)
    print vals

    vals = model.predict(data, batch_size=batch_size)
    savemat('predictions.mat', {'yhat': vals['output_t%d' % t_predict[0]], 'y': data['input_t%d' % (nt_in-1)]})



if __name__=='__main__':
    try:
        model = train()
        evaluate(model)
    except:
        ty, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
