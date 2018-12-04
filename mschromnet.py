"""Keras implementation of SSD."""

import keras.backend as K
from keras.layers import Activation
from keras.layers import AtrousConvolution1D
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers import GlobalAveragePooling1D
from keras.layers import Input
from keras.layers import MaxPooling1D
from keras.layers import UpSampling1D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import merge
from keras.layers import Reshape
from keras.layers import ZeroPadding1D
from keras.models import Model

from mschromnet_layers import PriorBox
from mschromnet_layers import Normalize

def Conv1DBNRelu(input, net, basename, cn):
    """CNN1D
    """
    net['conv' + basename] = Conv1D(cn, 3, padding='same', name='conv' + basename)(input)
    net['norm' + basename] = BatchNormalization(name='norm' + basename)(net['conv' + basename])
    net['relu' + basename] = Activation(activation='relu', name='relu' + basename)(net['norm' + basename])
    return net['relu' + basename]

def Conv1DBNSigmoid(input, net, basename, cn):
    """CNN1D
    """
    net['conv' + basename] = Conv1D(cn, 3, padding='same', name='conv' + basename)(input)
    net['norm' + basename] = BatchNormalization(name='norm' + basename)(net['conv' + basename])
    net['sigmoid' + basename] = Activation(activation='sigmoid', name='sigmoid' + basename)(net['norm' + basename])
    return net['sigmoid' + basename]

def MaxPool1D(input, net, basename):
    """MaxPool1D
    """
    net['pool' + basename] = MaxPooling1D(name='pool' + basename)(input)
    return net['pool' + basename]

def Upsample1D(input, net, basename):
    """Upsample1D
    """
    net['upsample' + basename] =  UpSampling1D(name='upsample' + basename)(input)
    return net['upsample' + basename]    

def MSChromNet(input_shape, num_classes=2):
    """SSD-like 1D architecture
    """
    net = {}
    # Block 1
    input_tensor = Input(shape=input_shape, name='input1')
    net['input'] = input_tensor
    net['reshape1'] = Reshape((input_shape[0],1), name='reshape1')(net['input'])

    x = Conv1DBNRelu(net['reshape1'], net, '1_1', 16)
    x = Conv1DBNRelu(x, net, '1_2', 16)
    x = MaxPool1D(x, net, '1')

    x = Conv1DBNRelu(x, net, '2_1', 32)
    x = Conv1DBNRelu(x, net, '2_2', 32)
    x = Conv1DBNRelu(x, net, '2_3', 32)
    net['conv2_3_norm'] = Normalize(20, name='conv2_3_norm')(x)
    num_priors = 3
    net['conv2_3_norm_mbox_loc'] = Conv1D(num_priors * 2, 3,
                        padding='same',
                        name='conv2_3_norm_mbox_loc')(net['conv2_3_norm'])
    net['conv2_3_norm_mbox_loc_flat'] = Flatten(name='conv2_3_norm_mbox_loc_flat')(net['conv2_3_norm_mbox_loc'])
    net['conv2_3_norm_mbox_conf'] = Conv1D(num_priors * num_classes, 3,
                        padding='same',
                        name='conv2_3_norm_mbox_conf')(net['conv2_3_norm'])
    net['conv2_3_norm_mbox_conf_flat'] = Flatten(name='conv2_3_norm_mbox_conf_flat')(net['conv2_3_norm_mbox_conf'])
    net['conv2_3_norm_mbox_priorbox'] = PriorBox(input_shape[0], 8.0,
                        aspect_ratios=[2],
                        variances=[0.1, 0.2],
                        name='conv2_3_norm_mbox_priorbox')(net['conv2_3_norm'])
    x = MaxPool1D(x, net, '2')

    x = Conv1DBNRelu(x, net, '3_1', 64)
    x = Conv1DBNRelu(x, net, '3_2', 64)
    x = Conv1DBNRelu(x, net, '3_3', 64)
    num_priors = 5
    net['conv3_3_mbox_loc'] = Conv1D(num_priors * 2, 3,
                        padding='same',
                        name='conv3_3_mbox_loc')(x)
    net['conv3_3_mbox_loc_flat'] = Flatten(name='conv3_3_mbox_loc_flat')(net['conv3_3_mbox_loc'])
    net['conv3_3_mbox_conf'] = Conv1D(num_priors * num_classes, 3,
                        padding='same',
                        name='conv3_3_mbox_conf')(x)
    net['conv3_3_mbox_conf_flat'] = Flatten(name='conv3_3_mbox_conf_flat')(net['conv3_3_mbox_conf'])
    net['conv3_3_mbox_priorbox'] = PriorBox(input_shape[0], 16.0,
                        aspect_ratios=[2, 3],
                        variances=[0.1, 0.2],
                        name='conv3_3_mbox_priorbox')(x)
    x = MaxPool1D(x, net, '3')

    x = Conv1DBNRelu(x, net, '4_1', 128)
    x = Conv1DBNRelu(x, net, '4_2', 128)
    x = Conv1DBNRelu(x, net, '4_3', 128)
    num_priors = 5
    net['conv4_3_mbox_loc'] = Conv1D(num_priors * 2, 3,
                        padding='same',
                        name='conv4_3_mbox_loc')(x)
    net['conv4_3_mbox_loc_flat'] = Flatten(name='conv4_3_mbox_loc_flat')(net['conv4_3_mbox_loc'])
    net['conv4_3_mbox_conf'] = Conv1D(num_priors * num_classes, 3,
                        padding='same',
                        name='conv4_3_mbox_conf')(x)
    net['conv4_3_mbox_conf_flat'] = Flatten(name='conv4_3_mbox_conf_flat')(net['conv4_3_mbox_conf'])
    net['conv4_3_mbox_priorbox'] = PriorBox(input_shape[0], 32.0,
                        aspect_ratios=[2, 3],
                        variances=[0.1, 0.2],
                        name='conv4_3_mbox_priorbox')(x)
    x = MaxPool1D(x, net, '4')

    x = Conv1DBNRelu(x, net, '5_1', 256)
    x = Conv1DBNRelu(x, net, '5_2', 256)
    x = Conv1DBNRelu(x, net, '5_3', 256)
    num_priors = 5
    net['conv5_3_mbox_loc'] = Conv1D(num_priors * 2, 3,
                        padding='same',
                        name='conv5_3_mbox_loc')(x)
    net['conv5_3_mbox_loc_flat'] = Flatten(name='conv5_3_mbox_loc_flat')(net['conv5_3_mbox_loc'])
    net['conv5_3_mbox_conf'] = Conv1D(num_priors * num_classes, 3,
                        padding='same',
                        name='conv5_3_mbox_conf')(x)
    net['conv5_3_mbox_conf_flat'] = Flatten(name='conv5_3_mbox_conf_flat')(net['conv5_3_mbox_conf'])
    net['conv5_3_mbox_priorbox'] = PriorBox(input_shape[0], 64.0,
                        aspect_ratios=[2, 3],
                        variances=[0.1, 0.2],
                        name='conv5_3_mbox_priorbox')(x)
    x = MaxPool1D(x, net, '5')

    x = Conv1DBNRelu(x, net, '6_1', 512)
    x = Conv1DBNRelu(x, net, '6_2', 512)
    x = Conv1DBNRelu(x, net, '6_3', 512)
    num_priors = 5
    net['conv6_3_mbox_loc'] = Conv1D(num_priors * 2, 3,
                        padding='same',
                        name='conv6_3_mbox_loc')(x)
    net['conv6_3_mbox_loc_flat'] = Flatten(name='conv6_3_mbox_loc_flat')(net['conv6_3_mbox_loc'])
    net['conv6_3_mbox_conf'] = Conv1D(num_priors * num_classes, 3,
                        padding='same',
                        name='conv6_3_mbox_conf')(x)
    net['conv6_3_mbox_conf_flat'] = Flatten(name='conv6_3_mbox_conf_flat')(net['conv6_3_mbox_conf'])
    net['conv6_3_mbox_priorbox'] = PriorBox(input_shape[0], 128.0,
                        aspect_ratios=[2, 3],
                        variances=[0.1, 0.2],
                        name='conv6_3_mbox_priorbox')(x)
    x = MaxPool1D(x, net, '6')

    x = Conv1DBNRelu(x, net, '7_1', 1024)
    x = Conv1DBNRelu(x, net, '7_2', 1024)
    x = Conv1DBNRelu(x, net, '7_3', 1024)
    num_priors = 5
    net['conv7_3_mbox_loc'] = Conv1D(num_priors * 2, 3,
                        padding='same',
                        name='conv7_3_mbox_loc')(x)
    net['conv7_3_mbox_loc_flat'] = Flatten(name='conv7_3_mbox_loc_flat')(net['conv7_3_mbox_loc'])
    net['conv7_3_mbox_conf'] = Conv1D(num_priors * num_classes, 3,
                        padding='same',
                        name='conv7_3_mbox_conf')(x)
    net['conv7_3_mbox_conf_flat'] = Flatten(name='conv7_3_mbox_conf_flat')(net['conv7_3_mbox_conf'])
    net['conv7_3_mbox_priorbox'] = PriorBox(input_shape[0], 256.0,
                        aspect_ratios=[2, 3],
                        variances=[0.1, 0.2],
                        name='conv7_3_mbox_priorbox')(x)
    x = MaxPool1D(x, net, '7')

    x = Conv1DBNRelu(x, net, '8_1', 1024) # ここは2048ではなく1024にして節約
    x = Conv1DBNRelu(x, net, '8_2', 1024)
    x = Conv1DBNRelu(x, net, '8_3', 1024)
    num_priors = 5
    net['conv8_3_mbox_loc'] = Conv1D(num_priors * 2, 3,
                        padding='same',
                        name='conv8_3_mbox_loc')(x)
    net['conv8_3_mbox_loc_flat'] = Flatten(name='conv8_3_mbox_loc_flat')(net['conv8_3_mbox_loc'])
    net['conv8_3_mbox_conf'] = Conv1D(num_priors * num_classes, 3,
                        padding='same',
                        name='conv8_3_mbox_conf')(x)
    net['conv8_3_mbox_conf_flat'] = Flatten(name='conv8_3_mbox_conf_flat')(net['conv8_3_mbox_conf'])
    net['conv8_3_mbox_priorbox'] = PriorBox(input_shape[0], 512.0,
                        aspect_ratios=[2, 3],
                        variances=[0.1, 0.2],
                        name='conv8_3_mbox_priorbox')(x)

    x = Upsample1D(x, net, '9')
    x = Conv1DBNRelu(x, net, '9_1', 1024)
    x = Conv1DBNRelu(x, net, '9_2', 1024)
    x = Conv1DBNRelu(x, net, '9_3', 1024)

    x = Upsample1D(x, net, '10')
    x = Conv1DBNRelu(x, net, '10_1', 512)
    x = Conv1DBNRelu(x, net, '10_2', 512)
    x = Conv1DBNRelu(x, net, '10_3', 512)

    x = Upsample1D(x, net, '11')
    x = Conv1DBNRelu(x, net, '11_1', 256)
    x = Conv1DBNRelu(x, net, '11_2', 256)
    x = Conv1DBNRelu(x, net, '11_3', 256)

    x = Upsample1D(x, net, '12')
    x = Conv1DBNRelu(x, net, '12_1', 128)
    x = Conv1DBNRelu(x, net, '12_2', 128)
    x = Conv1DBNRelu(x, net, '12_3', 128)

    x = Upsample1D(x, net, '13')
    x = Conv1DBNRelu(x, net, '13_1', 64)
    x = Conv1DBNRelu(x, net, '13_2', 64)
    x = Conv1DBNRelu(x, net, '13_3', 64)

    x = Upsample1D(x, net, '14')
    x = Conv1DBNRelu(x, net, '14_1', 32)
    x = Conv1DBNRelu(x, net, '14_2', 32)
    x = Conv1DBNRelu(x, net, '14_3', 32)

    x = Upsample1D(x, net, '15')
    x = Conv1DBNRelu(x, net, '15_1', 16)
    x = Conv1DBNRelu(x, net, '15_2', 16)
    x = Conv1DBNSigmoid(x, net, '15_3', 1)
    net['flatten15'] = Flatten()(x)

    # Gather Predictions

    net['mbox_loc'] = Concatenate(name='mbox_loc', axis=1)([net['conv2_3_norm_mbox_loc_flat'],
                            net['conv3_3_mbox_loc_flat'],
                            net['conv4_3_mbox_loc_flat'],
                            net['conv5_3_mbox_loc_flat'],
                            net['conv6_3_mbox_loc_flat'],
                            net['conv7_3_mbox_loc_flat'],
                            net['conv8_3_mbox_loc_flat']])
    net['mbox_conf'] = Concatenate(name='mbox_conf', axis=1)([net['conv2_3_norm_mbox_conf_flat'],
                            net['conv3_3_mbox_conf_flat'],
                            net['conv4_3_mbox_conf_flat'],
                            net['conv5_3_mbox_conf_flat'],
                            net['conv6_3_mbox_conf_flat'],
                            net['conv7_3_mbox_conf_flat'],
                            net['conv8_3_mbox_conf_flat']])
    net['mbox_priorbox'] = Concatenate(name='mbox_priorbox', axis=1)([net['conv2_3_norm_mbox_priorbox'],
                            net['conv3_3_mbox_priorbox'],
                            net['conv4_3_mbox_priorbox'],
                            net['conv5_3_mbox_priorbox'],
                            net['conv6_3_mbox_priorbox'],
                            net['conv7_3_mbox_priorbox'],
                            net['conv8_3_mbox_priorbox']])
    if hasattr(net['mbox_loc'], '_keras_shape'):
        num_boxes = net['mbox_loc']._keras_shape[-1] // 2
    elif hasattr(net['mbox_loc'], 'int_shape'):
        num_boxes = K.int_shape(net['mbox_loc'])[-1] // 2

    net['mbox_loc'] = Reshape((num_boxes, 2),
                              name='mbox_loc_final')(net['mbox_loc'])
    net['mbox_conf'] = Reshape((num_boxes, num_classes),
                               name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation('softmax',
                                  name='mbox_conf_final')(net['mbox_conf'])
    net['predictions'] = Concatenate(name='predictions', axis=2)([net['mbox_loc'],
                               net['mbox_conf'],
                               net['mbox_priorbox']])
    # # Dence 15
    # net['fc15_1'] = Flatten(name='fc15_1')(net['conv14_3'])
    # net['fc15_2'] = Dense(input_shape[0], activation='relu',
    #                                 name='fc15_2')(net['fc15_1'])
    # net['fc15_3'] = Dense(input_shape[0], activation='sigmoid',
    #                                 name='fc15_3')(net['fc15_2'])
    
    # Prediction
    # net['predictions'] = net['flatten15']    
    model = Model(net['input'], net['predictions'])
    return model

if __name__ == '__main__':
    input_shape = (1024, )
    mymodel = MSChromNet(input_shape)
    print(mymodel.summary())
