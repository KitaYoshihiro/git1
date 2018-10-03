"""Keras implementation of SSD."""

import keras.backend as K
from keras.layers import Activation
from keras.layers import AtrousConvolution1D
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Flatten
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

def MSChromNet(input_shape):
    """SSD-like 1D architecture
    """
    net = {}
    # Block 1
    input_tensor = Input(shape=input_shape)
    net['input'] = input_tensor
    net['reshape1'] = Reshape((input_shape[0],1))(net['input'])

    net['conv1_1'] = Conv1D(16, 3,
                                  padding='same',
                                  name='conv1_1')(net['reshape1'])
    net['norm1_1'] = BatchNormalization(name='norm1_1')(net['conv1_1'])
    net['relu1_1'] = Activation(activation='relu', name='relu1_1')(net['norm1_1'])

    net['conv1_2'] = Conv1D(16, 3,
                                  padding='same',
                                  name='conv1_2')(net['relu1_1'])
    net['norm1_2'] = BatchNormalization(name='norm1_2')(net['conv1_2'])
    net['relu1_2'] = Activation(activation='relu', name='relu1_2')(net['norm1_2'])

    # net['conv1_2'] = Conv1D(4, 3, activation='relu',
    #                               padding='same',
    #                               name='conv1_2')(net['relu1_1'])
    net['pool1'] = MaxPooling1D(name='pool1')(net['relu1_2'])
    # Block 2    
    net['conv2_1'] = Conv1D(32, 3,
                                  padding='same',
                                  name='conv2_1')(net['pool1'])
    net['norm2_1'] = BatchNormalization(name='norm2_1')(net['conv2_1'])
    net['relu2_1'] = Activation(activation='relu', name='relu2_1')(net['norm2_1'])

    net['conv2_2'] = Conv1D(32, 3,
                                  padding='same',
                                  name='conv2_2')(net['relu2_1'])
    net['norm2_2'] = BatchNormalization(name='norm2_2')(net['conv2_2'])
    net['relu2_2'] = Activation(activation='relu', name='relu2_2')(net['norm2_2'])

    net['pool2'] = MaxPooling1D(name='pool2')(net['relu2_2'])
    net['upsample2'] = UpSampling1D()(net['relu2_2'])

    # Block 3
    net['conv3_1'] = Conv1D(64, 3, activation='relu',
                                   padding='same',
                                   name='conv3_1')(net['pool2'])
    net['conv3_2'] = Conv1D(64, 3, activation='relu',
                                   padding='same',
                                   name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = Conv1D(64, 3, activation='relu',
                                   padding='same',
                                   name='conv3_3')(net['conv3_2'])
    net['pool3'] = MaxPooling1D(name='pool3')(net['conv3_3'])
    # Block 4
    net['conv4_1'] = Conv1D(128, 3, activation='relu',
                                   padding='same',
                                   name='conv4_1')(net['pool3'])
    net['conv4_2'] = Conv1D(128, 3, activation='relu',
                                   padding='same',
                                   name='conv4_2')(net['conv4_1'])
    net['conv4_3'] = Conv1D(128, 3, activation='relu',
                                   padding='same',
                                   name='conv4_3')(net['conv4_2'])
    net['pool4'] = MaxPooling1D(name='pool4')(net['conv4_3'])
    net['upsample4'] = UpSampling1D()(net['conv4_3'])

    # Block 5
    net['conv5_1'] = Conv1D(256, 3, activation='relu',
                                   padding='same',
                                   name='conv5_1')(net['pool4'])
    net['conv5_2'] = Conv1D(256, 3, activation='relu',
                                   padding='same',
                                   name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = Conv1D(256, 3, activation='relu',
                                   padding='same',
                                   name='conv5_3')(net['conv5_2'])
    net['pool5'] = MaxPooling1D(name='pool5')(net['conv5_3'])
    # Block 6
    net['conv6_1'] = Conv1D(512, 3, activation='relu',
                                    padding='same',
                                    name='conv6_1')(net['pool5'])
    net['conv6_2'] = Conv1D(512, 3, activation='relu',
                                    padding='same',
                                    name='conv6_2')(net['conv6_1'])
    net['conv6_3'] = Conv1D(512, 3, activation='relu',
                                    padding='same',
                                    name='conv6_3')(net['conv6_2'])
    net['pool6'] = MaxPooling1D(name='pool6')(net['conv6_3'])
    # Block 7
    net['conv7_1'] = Conv1D(1024, 3, activation='relu',
                                    padding='same',
                                    name='conv7_1')(net['pool6'])
    net['conv7_2'] = Conv1D(1024, 3, activation='relu',
                                    padding='same',
                                    name='conv7_2')(net['conv7_1'])
    net['conv7_3'] = Conv1D(1024, 3, activation='relu',
                                    padding='same',
                                    name='conv7_3')(net['conv7_2'])
    net['pool7'] = MaxPooling1D(name='pool7')(net['conv7_3'])

    # Block 8
    net['conv8_1'] = Conv1D(2048, 3, activation='relu',
                                    padding='same',
                                    name='conv8_1')(net['pool7'])
    net['conv8_2'] = Conv1D(2048, 3, activation='relu',
                                    padding='same',
                                    name='conv8_2')(net['conv8_1'])
    net['conv8_3'] = Conv1D(2048, 3, activation='relu',
                                    padding='same',
                                    name='conv8_3')(net['conv8_2'])
    net['upsample8'] = UpSampling1D(name='upsample8')(net['conv8_3'])

    # Block 9
    net['conv9_1'] = Conv1D(1024, 3, activation='relu',
                                    padding='same',
                                    name='conv9_1')(net['upsample8'])
    net['conv9_2'] = Conv1D(1024, 3, activation='relu',
                                    padding='same',
                                    name='conv9_2')(net['conv9_1'])
    net['conv9_3'] = Conv1D(1024, 3, activation='relu',
                                    padding='same',
                                    name='conv9_3')(net['conv9_2'])
    net['upsample9'] = UpSampling1D(name='upsample9')(net['conv9_3'])
    # Block 10
    net['conv10_1'] = Conv1D(512, 3, activation='relu',
                                    padding='same',
                                    name='conv10_1')(net['upsample9'])
    net['conv10_2'] = Conv1D(512, 3, activation='relu',
                                    padding='same',
                                    name='conv10_2')(net['conv10_1'])
    net['conv10_3'] = Conv1D(512, 3, activation='relu',
                                    padding='same',
                                    name='conv10_3')(net['conv10_2'])
    net['upsample10'] = UpSampling1D(name='upsample10')(net['conv10_3'])
    # Block 11
    net['conv11_1'] = Conv1D(256, 3, activation='relu',
                                    padding='same',
                                    name='conv11_1')(net['upsample10'])
    net['conv11_2'] = Conv1D(256, 3, activation='relu',
                                    padding='same',
                                    name='conv11_2')(net['conv11_1'])
    net['conv11_3'] = Conv1D(256, 3, activation='relu',
                                    padding='same',
                                    name='conv11_3')(net['conv11_2'])
    net['upsample11'] = UpSampling1D(name='upsample11')(net['conv11_3'])
    # Block 12
    net['conv12_1'] = Conv1D(128, 3, activation='relu',
                                    padding='same',
                                    name='conv12_1')(net['upsample11'])
    net['conv12_2'] = Conv1D(128, 3, activation='relu',
                                    padding='same',
                                    name='conv12_2')(net['conv12_1'])
    net['conv12_3'] = Conv1D(128, 3, activation='relu',
                                    padding='same',
                                    name='conv12_3')(net['conv12_2'])
    net['upsample12'] = UpSampling1D(name='upsample12')(net['conv12_3'])
    # Block 13
    net['conv13_1'] = Conv1D(64, 3, activation='relu',
                                    padding='same',
                                    name='conv13_1')(net['upsample12'])
    # net['conv13_1'] = Conv1D(16, 3, activation='relu',
    #                                 padding='same',
    #                                 name='conv13_1')(net['upsample4'])
    net['conv13_2'] = Conv1D(64, 3, activation='relu',
                                    padding='same',
                                    name='conv13_2')(net['conv13_1'])
    net['conv13_3'] = Conv1D(64, 3, activation='relu',
                                    padding='same',
                                    name='conv13_3')(net['conv13_2'])
    net['upsample13'] = UpSampling1D(name='upsample13')(net['conv13_3'])
    # Block 14
    net['conv14_1'] = Conv1D(32, 3, activation='relu',
                                    padding='same',
                                    name='conv14_1')(net['upsample13'])
    net['conv14_2'] = Conv1D(32, 3, activation='relu',
                                    padding='same',
                                    name='conv14_2')(net['conv14_1'])
    net['conv14_3'] = Conv1D(32, 3, activation='relu',
                                    padding='same',
                                    name='conv14_3')(net['conv14_2'])
    net['upsample14'] = UpSampling1D(name='upsample14')(net['conv14_3'])
    # Block 15
    net['conv15_1'] = Conv1D(16, 3,
                                    padding='same',
                                    name='conv15_1')(net['upsample14'])
    net['norm15_1'] = BatchNormalization(name='norm15_1')(net['conv15_1'])
    net['relu15_1'] = Activation(activation='relu', name='relu15_1')(net['norm15_1'])                    
    # net['conv15_1'] = Conv1D(4, 3, activation='relu',
    #                                 padding='same',
    #                                 name='conv15_1')(net['upsample14'])
    net['conv15_2'] = Conv1D(16, 3,
                                    padding='same',
                                    name='conv15_2')(net['relu15_1'])
    net['norm15_2'] = BatchNormalization(name='norm15_2')(net['conv15_2'])
    net['relu15_2'] = Activation(activation='relu', name='relu15_2')(net['norm15_2'])

    net['conv15_3'] = Conv1D(1, 3,
                                    padding='same',
                                    name='conv15_3')(net['relu15_2'])
    net['norm15_3'] = BatchNormalization(name='norm15_3')(net['conv15_3'])
    net['relu15_3'] = Activation(activation='sigmoid', name='relu15_3')(net['norm15_3'])

    net['flatten15_4'] = Flatten()(net['relu15_3'])

    # # Dence 15
    # net['fc15_1'] = Flatten(name='fc15_1')(net['conv14_3'])
    # net['fc15_2'] = Dense(input_shape[0], activation='relu',
    #                                 name='fc15_2')(net['fc15_1'])
    # net['fc15_3'] = Dense(input_shape[0], activation='sigmoid',
    #                                 name='fc15_3')(net['fc15_2'])
    
    # Prediction
    net['predictions'] = net['flatten15_4']    
    model = Model(net['input'], net['predictions'])
    return model

if __name__ == '__main__':
    input_shape = (1024, )
    mymodel = MSChromNet(input_shape)
    print(mymodel.summary())
