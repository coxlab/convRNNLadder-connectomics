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

    def build(self):
        loss = {}
        loss_weights = {}
        for i,t in enumerate(self.config.t_predict):
            loss['output_t%d' % t] = self.config.loss
            if hasattr(self.config, 'loss_weights'):
                loss_weights['output_t%d' % t] = self.config.loss_weights[i]
            else:
                loss_weights['output_t%d' % t] = 1.0

        self.compile(loss=loss, optimizer=self.config.optimizer, loss_weights=loss_weights)

    def format_data(self, X, Y=None):
        data = {}
        n = X.shape[0]

        for t in range(self.config.nt_in):
            data['input_t%d' % t] = X[:,t]

        for l in range(self.config.n_modules):
            data['H_l%d_t-1' % l] = np.zeros((n - self.config.nt_in + 1, self.config.stack_sizes[l], self.config.input_shape[1] // 2**(l+1), self.config.input_shape[2] // 2**(l+1))).astype(np.float32)
            data['C_l%d_t-1' % l] = np.copy(data['H_l%d_t-1' % l])

        if Y is not None:
            for t in self.config.t_predict:
                data['output_t%d' % t] = Y[:,t]

    def format_predictions(self, data):
        for i,t in enumerate(self.config.t_predict):
            Xt = data['output_t%d' % t]
            if i==0:
                X = np.zeros( (Xt.shape[0], len(self.config.t_predict) + Xt.shape[1:])).astype(np.float32)
            X[:,i] = Xt
        return X

    def initialize(self):
        # initialize hidden states
        for l in range(self.config.n_modules):
            self.add_input(name='H_l%d_t-1' % l, input_shape=(self.config.stack_sizes[l], self.config.input_shape[0] // 2**(l+1), self.config.input_shape[1] // 2**(l+1)))
            self.add_input(name='C_l%d_t-1' % l, input_shape=(self.config.stack_sizes[l], self.config.input_shape[0] // 2**(l+1), self.config.input_shape[1] // 2**(l+1)))

        for t in range(self.config.nt_in):
            self.add_input(name='input_t%d' % t, input_shape=(1, self.config.input_shape[0], self.config.input_shape[1]))

            # add first conv layer
            if t == 0:
                trainable = True
                shared_layer = None
            else:
                trainable = False
                shared_layer = self.nodes['conv0pre_l-1_t0']
            layer = Convolution2D(self.config.stack_sizes[0], 5, 5, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layer)
            self.add_node(layer, name='conv0pre_l-1_t%d' % t, input='input_t%d' % t)
            self.add_node(AveragePooling2D(), name='conv0_l-1_t%d' % t, input='conv0pre_l-1_t%d' % t)

            for l in range(self.config.n_modules):
                layer_names = ['conv%d_l%d_t%d' % (i, l, t) for i in range(4)]
                if t == 0:
                    trainable = True
                    shared_layers = [None for _ in range(4)]
                else:
                    trainable = False
                    shared_layers = [self.nodes['conv%d_l%d_t0' % (i, l)] for i in range(4)]

                if l==0:
                    module_input = 'conv0_l-1_t%d' % t
                    module_input_upchannel = 'conv0_l-1_t%d' % t
                else:
                    module_input = 'H_l%d_t%d' % (l-1, t)
                    module_input_upchannel = 'Hupchannel_l%d_t%d' % (l-1, t)
                    if t == 0:
                        shared_l = None
                    else:
                        shared_l = self.nodes['Hupchannel_l%d_t0' % (l-1)]
                    layer = Convolution2D(self.config.stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_l)
                    self.add_node(layer, name='Hupchannel_l%d_t%d' % (l-1, t), input=module_input)


                layer = Convolution2D(self.config.stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[0])
                self.add_node(layer, name=layer_names[0], input=module_input)
                layer = Convolution2D(self.config.stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[1])
                self.add_node(layer, name=layer_names[1], input=layer_names[0])
                self.add_node(Activation('linear'), name='res0_l%d_t%d' % (l, t), inputs=[module_input_upchannel, layer_names[1]], merge_mode='sum')

                layer = Convolution2D(self.config.stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[2])
                self.add_node(layer, name=layer_names[2], input='res0_l%d_t%d' % (l, t))
                layer = Convolution2D(self.config.stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[3])
                self.add_node(layer, name=layer_names[3], input=layer_names[2])
                if l > 0:
                    self.add_node(AveragePooling2D(), name='res1_l%d_t%d' % (l, t), inputs=['res0_l%d_t%d' % (l, t), layer_names[3]], merge_mode='sum')
                else:
                    self.add_node(Activation('linear'), name='res1_l%d_t%d' % (l, t), inputs=['res0_l%d_t%d' % (l, t), layer_names[3]], merge_mode='sum')


                if t==0:
                    shared_layers = [None for _ in range(4)]
                else:
                    shared_layers = [self.nodes['I_l%d_t0' % l], self.nodes['F_l%d_t0' % l], self.nodes['O_l%d_t0' % l], self.nodes['C1_l%d_t0' % l] ]

                layer = Convolution2D(self.config.stack_sizes[l], 3, 3, border_mode='same', activation='hard_sigmoid', trainable=trainable, shared_layer=shared_layers[0])
                self.add_node(layer, name='I_l%d_t%d' % (l, t), inputs=['res1_l%d_t%d' % (l, t), 'H_l%d_t%d' % (l, t-1)], merge_mode='concat', concat_axis=-3)
                layer = Convolution2D(self.config.stack_sizes[l], 3, 3, border_mode='same', activation='hard_sigmoid', trainable=trainable, shared_layer=shared_layers[1])
                self.add_node(layer, name='F_l%d_t%d' % (l, t), inputs=['res1_l%d_t%d' % (l, t), 'H_l%d_t%d' % (l, t-1)], merge_mode='concat', concat_axis=-3)
                layer = Convolution2D(self.config.stack_sizes[l], 3, 3, border_mode='same', activation='hard_sigmoid', trainable=trainable, shared_layer=shared_layers[2])
                self.add_node(layer, name='O_l%d_t%d' % (l, t), inputs=['res1_l%d_t%d' % (l, t), 'H_l%d_t%d' % (l, t-1)], merge_mode='concat', concat_axis=-3)

                layer = Convolution2D(self.config.stack_sizes[l], 3, 3, border_mode='same', activation='tanh', trainable=trainable, shared_layer=shared_layers[3])
                self.add_node(layer, name='C1_l%d_t%d' % (l, t), inputs=['res1_l%d_t%d' % (l, t), 'H_l%d_t%d' % (l, t-1)], merge_mode='concat', concat_axis=-3)

                self.add_node(Activation('linear'), name='C2_l%d_t%d' % (l, t), inputs=['I_l%d_t%d' % (l, t), 'C1_l%d_t%d' % (l, t)], merge_mode='mul')
                self.add_node(Activation('linear'), name='C3_l%d_t%d' % (l, t), inputs=['F_l%d_t%d' % (l, t), 'C_l%d_t%d' % (l, t-1)], merge_mode='mul')
                self.add_node(Activation('linear'), name='C_l%d_t%d' % (l, t), inputs=['C3_l%d_t%d' % (l, t), 'C2_l%d_t%d' % (l, t)], merge_mode='sum')
                self.add_node(Activation('tanh'), name='Ct_l%d_t%d' % (l, t), input='C_l%d_t%d' % (l, t))

                self.add_node(Activation('linear'), name='H_l%d_t%d' % (l, t), inputs=['O_l%d_t%d' % (l, t), 'Ct_l%d_t%d' % (l, t)], merge_mode='mul')

            if t in self.config.t_predict:
                for l in range(self.config.n_modules-1, -1, -1):
                    layer_names = ['deconv%d_l%d_t%d' % (i, l, t) for i in range(4)]
                    if t == t_predict[0]:
                        trainable = True
                        shared_layers = [None for _ in range(4)]
                    else:
                        trainable = False
                        shared_layers = [self.nodes['deconv%d_l%d_t%d' % (i, l, t_predict[0])] for i in range(4)]


                    if l==n_modules-1:
                        module_input = 'Hup_l%d_t%d' % (l, t)
                    else:
                        module_input = 'comb_l%d_t%d' % (l, t)


                    self.add_node(UpSampling2D(), name='Hup_l%d_t%d' % (l, t), input='H_l%d_t%d' % (l, t))
                    if l<n_modules-1:
                       self.add_node(UpSampling2D(), name='deres1up_l%d_t%d' % (l+1, t), input='deres1_l%d_t%d' % (l+1, t))

                    if l<n_modules-1:
                        #self.add_node(Activation('linear'), name='prod_l%d_t%d' % (l, t), inputs=['Hup_l%d_t%d' % (l, t), 'deres1_l%d_t%d' % (l+1, t)], merge_mode='mul')
                        if t == t_predict[0]:
                            shared_l = None
                            tr = True
                        else:
                            shared_l = self.nodes['comb_l%d_t%d' % (l, t_predict[0])]
                            tr = False
                        layer = Convolution2D(self.config.stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=tr, shared_layer=shared_l)
                        self.add_node(layer, name='comb_l%d_t%d' % (l, t), inputs=['Hup_l%d_t%d' % (l, t), 'deres1up_l%d_t%d' % (l+1, t)], merge_mode='concat', concat_axis=-3)


                    layer = Convolution2D(self.config.stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[0])
                    self.add_node(layer, name=layer_names[0], input=module_input)
                    layer = Convolution2D(self.config.stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[1])
                    self.add_node(layer, name=layer_names[1], input=layer_names[0])
                    self.add_node(Activation('linear'), name='deres0_l%d_t%d' % (l, t), inputs=[module_input, layer_names[1]], merge_mode='sum')

                    layer = Convolution2D(self.config.stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[2])
                    self.add_node(layer, name=layer_names[2], input='deres0_l%d_t%d' % (l, t))
                    layer = Convolution2D(self.config.stack_sizes[l], 3, 3, border_mode='same', activation='relu', trainable=trainable, shared_layer=shared_layers[3])
                    self.add_node(layer, name=layer_names[3], input=layer_names[2])

                    self.add_node(Activation('linear'), name='deres1_l%d_t%d' % (l, t), inputs=['deres0_l%d_t%d' % (l, t), layer_names[3]], merge_mode='sum')

                    if l==0:
                        self.add_node(Convolution2D(1, 1, 1), name='output_t%d' % t, input='deres1_l%d_t%d' % (l, t), create_output=True)
