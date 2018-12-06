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

def Concat(input, input2, net, basename):
    """Concat
    """
    net['concat' + basename] =  Concatenate(name='concat' + basename, axis=2)([input, input2])
    return net['concat' + basename] 

def UNet_Builder(input, net, initial_layer_id, structure, depth=0):
    """ building U-net
    # input: input keras tensor
    # net: list for network layers (keras tensors)
    # initial_layer_id: int value for starting layer number (for example, initial_layer_id=2)
    # structure: 2D-array of channels (example: structure = [[16,16],[32,32,32],[64,64,64]])
    """
    structure_length = len(structure)
    if depth > structure_length or depth <= 1:
        depth = structure_length
    
    if depth != structure_length:
        depth_label = '_d' + str(depth)
    else:
        depth_label = ''

    while len(structure) > depth:
        structure.pop()

    channels = structure.pop(0)
    x = input
    subid = 1
    for channel in channels:
        x = Conv1DBNRelu(x, net, '_f_'+str(initial_layer_id)+'_'+str(subid), channel)
        subid += 1
    if len(structure) == 0:
        return x
    xx = x
    if len(structure) > 0:
        x = MaxPool1D(x, net, '_f_'+str(initial_layer_id))
        initial_layer_id +=1
        x = UNet_Builder(x, net, initial_layer_id, structure)
        initial_layer_id -=1
        x = Upsample1D(x, net, '_r_'+str(initial_layer_id)+depth_label)
        x = Concat(xx, x, net, '_r_'+str(initial_layer_id)+depth_label) 
    subid = 1
    for channel in reversed(channels):
        x = Conv1DBNRelu(x, net, '_r_'+str(initial_layer_id)+'_'+str(subid)+depth_label, channel)
        subid += 1        
    return x

def MSChromUNet(input_shape, depth=0, num_classes=2):
    """SSD-like 1D architecture
    """
    net = {}
    # input
    input_tensor = Input(shape=input_shape, name='input1')
    net['input'] = input_tensor
    net['reshape1'] = Reshape((input_shape[0],1), name='reshape1')(net['input'])
    x = net['reshape1']
    structure = [[16,16],[32,32,32],[64,64,64],[128,128,128],
                [256,256,256],[512,512,512],[1024,1024,1024],[1024,1024,1024]]
    x = UNet_Builder(x, net, 1, structure, depth)
    x = Conv1DBNSigmoid(x, net, '_autoencoder', 1)
    net['autoencoder_flatten'] = Flatten(name='autoencoder_flatten')(x)

    # Prediction
    net['predictions'] = net['autoencoder_flatten']
    model = Model(net['input'], net['predictions'])
    return model

if __name__ == '__main__':
    input_shape = (1024, )
    mymodel = MSChromUNet(input_shape)
    print(mymodel.summary())
