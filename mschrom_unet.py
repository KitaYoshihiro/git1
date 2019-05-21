"""Keras implementation of SSD."""
import numpy as np
import keras.backend as K
if K.backend() == 'cntk':
    import cntk as C
import keras
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
from keras.layers import merge
from keras.layers import Reshape
from keras.layers import ZeroPadding1D
from keras.models import Model

from mschromnet_layers import PriorBox
from mschromnet_layers import Normalize
from mschromnet_layers import MagnifyAndClip
from mschromnet_layers import LogTransform

from mschromnet_utils import BBoxUtility
from gdrivegenerator import GdriveGenerator
import pickle

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

def UNet_Builder(input, net, initial_layer_id, structure, depth=20, u_net=True, autoencoder=False):
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
    
    if u_net:
        depth_label += 'u'

    while len(structure) > depth:
        structure.pop()

    channels = structure.pop(0)
    x = input
    subid = 1
    for channel in channels:
        x = Conv1DBNRelu(x, net, '_f_'+str(initial_layer_id)+'_'+str(subid), channel)
        subid += 1
    # if len(structure) == 0:
    #    return x # 最下層はここでリターン！
    # xx = x
    if len(structure) > 0:
        xx = x
        x = MaxPool1D(x, net, '_f_'+str(initial_layer_id))
        initial_layer_id +=1
        x = UNet_Builder(x, net, initial_layer_id, structure, u_net=u_net, autoencoder=autoencoder)
        initial_layer_id -=1
        x = Upsample1D(x, net, '_r_'+str(initial_layer_id)+depth_label)
        if u_net:
            x = Concat(xx, x, net, '_r_'+str(initial_layer_id)+depth_label) 
        subid = 1
        for channel in reversed(channels):
            x = Conv1DBNRelu(x, net, '_r_'+str(initial_layer_id)+'_'+str(subid)+depth_label, channel)
            subid += 1
    # ConvBNReLUを複数回済ませてリターンする直前（返値はその後Upsampleされる）のこの位置にLoc, Conf, Priorレイヤを設置
    if not autoencoder:
        num_priors = 5
        channels_num = x._keras_shape[1] # レイヤの大きさを拾う（8～1024）
        min_width = 1024//channels_num # それをPriorboxのmin_widthに使う
        net['L'+str(initial_layer_id)+'_mbox_loc'] = Conv1D(num_priors * 2, 3,
                            padding='same',
                            name='L'+str(initial_layer_id)+'_mbox_loc')(x)
        net['L'+str(initial_layer_id)+'_mbox_loc_flat'] = Flatten(name='L'+str(initial_layer_id)+'_mbox_loc_flat')(net['L'+str(initial_layer_id)+'_mbox_loc'])
        net['L'+str(initial_layer_id)+'_mbox_conf'] = Conv1D(num_priors * 2, 3,
                            padding='same',
                            name='L'+str(initial_layer_id)+'_mbox_conf')(x)
        net['L'+str(initial_layer_id)+'_mbox_conf_flat'] = Flatten(name='L'+str(initial_layer_id)+'_mbox_conf_flat')(net['L'+str(initial_layer_id)+'_mbox_conf'])
        net['L'+str(initial_layer_id)+'_mbox_priorbox'] = PriorBox(net['input']._keras_shape[1], min_width,
                            aspect_ratios=[2, 3],
                            variances=[0.1, 0.2],
                            name='L'+str(initial_layer_id)+'_mbox_priorbox')(x)
    return x

def MSChromUNet(input_shape, depth=18, u_net=True, autoencoder=False, magnify=False, logtransform=False, num_classes=2):
    """SSD-like 1D architecture
    """
    net = {}
    # input
    input_tensor = Input(shape=input_shape, name='input1')
    net['input'] = input_tensor
    net['reshape1'] = Reshape((input_shape[0],1), name='reshape1')(net['input'])
    x = net['reshape1']
    if magnify:
        net['magnify1'] = MagnifyAndClip(name='magnify1')(x)
        x = net['magnify1']
        net['conv0'] = Conv1D(64, 1, padding='same', name='conv0')(x)
        x = net['conv0']
    if logtransform:
        net['logtransform1'] = LogTransform(name='logtransform1', scale=7)(x)
        x = net['logtransform1']
    # structure = [[64,64],[64,64,64],[64,64,64],[128,128,128],
    #             [256,256,256],[512,512,512],[1024,1024,1024],[1024,1024,1024]]#,
    #             #[1024,1024,1024],[1024,1024,1024]]
    structure = [[64,64],[64,64,64],[64,64,64],[128,128,128],
                [256,256,256],[512,512,512],[1024,1024,1024],[1024,1024,1024],
                [1024,1024,1024],[1024,1024,1024],[1024,1024,1024],[1024,1024,1024],
                [1024,1024,1024],[1024,1024,1024],[1024,1024,1024],[1024,1024,1024],
                [1024,1024,1024],[1024,1024,1024],[1024,1024,1024],[1024,1024,1024],
                [1024,1024,1024],[1024,1024,1024],[1024,1024,1024],[1024,1024,1024]]
    # structure = [[64,64],[64,64,64],[64,64,64],[128,128,128],
    #             [128,128,128],[256,256,256],[256,256,256],[512,512,512],
    #             [512,512,512],[512,512,512]]
    x = UNet_Builder(x, net, 1, structure, depth, u_net=u_net, autoencoder=autoencoder)

    # Autoencoder
    x = Conv1DBNSigmoid(x, net, '_autoencoder', 1)
    net['autoencoder_flatten'] = Flatten(name='autoencoder_flatten')(x)

    # Gather Predictions
    if not autoencoder:
        net['mbox_loc'] = Concatenate(name='mbox_loc', axis=1)([
                                # net['L10_mbox_loc_flat'],
                                # net['L9_mbox_loc_flat'],
                                net['L8_mbox_loc_flat'],
                                net['L7_mbox_loc_flat'],
                                net['L6_mbox_loc_flat'],
                                net['L5_mbox_loc_flat'],
                                net['L4_mbox_loc_flat'],
                                net['L3_mbox_loc_flat'],
                                net['L2_mbox_loc_flat'],
                                net['L1_mbox_loc_flat']
                                ])
        net['mbox_conf'] = Concatenate(name='mbox_conf', axis=1)([
                                # net['L10_mbox_conf_flat'],
                                # net['L9_mbox_conf_flat'],
                                net['L8_mbox_conf_flat'],
                                net['L7_mbox_conf_flat'],
                                net['L6_mbox_conf_flat'],
                                net['L5_mbox_conf_flat'],
                                net['L4_mbox_conf_flat'],
                                net['L3_mbox_conf_flat'],
                                net['L2_mbox_conf_flat'],
                                net['L1_mbox_conf_flat']
                                ])
        net['mbox_priorbox'] = Concatenate(name='mbox_priorbox', axis=1)([
                                # net['L10_mbox_priorbox'],
                                # net['L9_mbox_priorbox'],
                                net['L8_mbox_priorbox'],
                                net['L7_mbox_priorbox'],
                                net['L6_mbox_priorbox'],
                                net['L5_mbox_priorbox'],
                                net['L4_mbox_priorbox'],
                                net['L3_mbox_priorbox'],
                                net['L2_mbox_priorbox'],
                                net['L1_mbox_priorbox']
                                ])
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

        # for adjustment for CNTK
        #net['mbox_priorbox'] = Reshape((-1, 4), name = 'mbox_priorbox_final')(net['mbox_priorbox'])
        # for tensorflow
        net['predictions'] = Concatenate(name='predictions', axis=2)([net['mbox_loc'],
                                net['mbox_conf']]) #,
                                #net['mbox_priorbox']])
        # # for CNTK
        # net['predictions'] = Concatenate(name='predictions', axis=1)([net['mbox_loc'],
        #                         net['mbox_conf'],
        #                         net['mbox_priorbox']])
    else:
        net['predictions'] = net['autoencoder_flatten']

    model = Model(net['input'], net['predictions'])
    return model

if __name__ == '__main__':
    input_shape = (1048576, )
    mymodel = MSChromUNet(input_shape, 18, u_net=False, autoencoder=True, magnify=False, logtransform=False)
    # for L in mymodel.layers:
    #     if 'conv' in L.name:
    #         print(L.name)
    #         L.trainable = False
    print(mymodel.summary())

    # ksess = K.get_session()
    # print(ksess)
    # K.set_learning_phase(0)
    # graph = ksess.graph
    # kgraph = graph.as_graph_def()
    # print(kgraph)

    # load weights
    mymodel.load_weights('../wt.e014-0.41717_d8u_detector_allweights_batch32.hdf5', by_name=True)
    
    # mymodel.save('../my_tf_model.h5')
    C.combine(mymodel.outputs).save('../unet_predictor_d8_wo_d123456out.dnn')
    
    # load priors    
    with open('mschrom_unet_priors.pkl', mode='rb') as f:
        priors = pickle.load(f)
    # load data
    with open('../sharp_peaks.pickle', mode='rb') as f:
        tr = pickle.load(f)
    num_classes = 2
    bbox_util = BBoxUtility(num_classes=num_classes, priors=priors)
    gen = GdriveGenerator(bbox_util=bbox_util, batch_size=128, train_data=tr, validate_data=tr)
    g = gen.generate(train=False, autoencoder=False)
    chrom, gt = next(g)

    import winmltools
    # model = winmltools.convert_keras(mymodel, 7, name='mymodel')
    model = winmltools.convert_tensorflow(graph, 7)

    def schedule(epoch, decay=0.9):
        return base_lr * decay**(epoch)
    base_lr = 1e-3
    momentum = 0.9
    decay = 0.0005
    sgd = keras.optimizers.SGD(lr=base_lr, decay=decay, momentum=momentum, nesterov=True)

    # from mschromnet_training import MultiboxLoss 
    # mymodel.compile(optimizer=sgd, loss=MultiboxLoss(num_classes, neg_pos_ratio=3.0).compute_loss)
    # gen.batch_size = 4
    # import multiprocessing
    # process_count = multiprocessing.cpu_count() - 1
    # nb_epoch = 300
    # callbacks = [keras.callbacks.LearningRateScheduler(schedule)]
    # history = mymodel.fit_generator(gen.generate(True, autoencoder=False), steps_per_epoch=32, epochs=nb_epoch, verbose=1,
    #                           callbacks=callbacks, validation_data=gen.generate(False, autoencoder=False), validation_steps=2,
    #                           workers=process_count, use_multiprocessing=True)

    predictions = mymodel.predict(chrom, batch_size=1, verbose=1)
    # results = bbox_util.detection_out(predictions)
    results2 = np.array(results)
    print(results2.shape)
