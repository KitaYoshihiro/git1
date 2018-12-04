"""Special layers for MSChromNet"""

import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

import pickle
import os

class Normalize(Layer):
    """Normalization layer as described in ParseNet paper.

    # Arguments
        scale: Default feature scale.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        Same as input

    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf

    #TODO
        Add possibility to have one scale for all features.
    """
    def __init__(self, scale, **kwargs):
        if K.image_dim_ordering() == 'tf':
            self.axis = 2 # 3から2に変更、多分これで良い？
        else:
            self.axis = 1
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = K.variable(init_gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        output *= self.gamma
        return output


class PriorBox(Layer):
    """Generate the prior boxes of designated sizes and aspect ratios.

    # Arguments
        data_length: Length of the input data.
        min_width: Minimum width of box size in datapoints.
        max_width: Maximum width of box size in datapoints.
        aspect_ratios: List of aspect ratios of boxes.
        flip: Whether to consider reverse aspect ratios.
        variances: List of variances for x, y, w, h.
        clip: Whether to clip the prior's coordinates
            such that they are within [0, 1].

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        3D tensor with shape:
        (samples, num_boxes, 8)

    # References
        https://arxiv.org/abs/1512.02325

    #TODO
        Add possibility not to have variances.
        Add Theano support
    """
    def __init__(self, data_length, min_width, max_width=None, aspect_ratios=None,
                 flip=True, variances=[0.1], clip=True, **kwargs):
        if K.image_dim_ordering() == 'tf':
            self.waxis = 1
            # self.waxis = 2
            # self.haxis = 1
        else:
            self.waxis = 2
            # self.waxis = 3
            # self.haxis = 2
        self.data_length = data_length
        if min_width <= 0:
            raise Exception('min_width must be positive.')
        self.min_width = min_width
        self.max_width = max_width
        self.aspect_ratios = [1.0]
        if max_width:
            if max_width < min_width:
                raise Exception('max_width must be greater than min_width.')
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = True
        super(PriorBox, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        num_priors_ = len(self.aspect_ratios)
        layer_width = input_shape[self.waxis]
        # layer_height = input_shape[self.haxis]
        num_boxes = num_priors_ * layer_width # * layer_height
        # return (input_shape[0], num_boxes, 8)
        return (input_shape[0], num_boxes, 4) # 1D なので x_min, x_max, variance_width, variance_scaleの４つ

    def call(self, x, mask=None):
        if hasattr(x, '_keras_shape'):
            input_shape = x._keras_shape
        elif hasattr(K, 'int_shape'):
            input_shape = K.int_shape(x)
        layer_width = input_shape[self.waxis]
        # layer_height = input_shape[self.haxis]
        data_length = self.data_length
        # img_height = self.img_size[1]
        # define prior boxes shapes
        box_widths = []
        # box_heights = []
        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_width)
                # box_heights.append(self.min_width)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_width * self.max_width))
                # box_heights.append(np.sqrt(self.min_width * self.max_width))
            elif ar != 1:
                box_widths.append(self.min_width * np.sqrt(ar))
                # box_heights.append(self.min_size / np.sqrt(ar))
        box_widths = 0.5 * np.array(box_widths)
        # box_heights = 0.5 * np.array(box_heights)
        # define centers of prior boxes
        step_x = data_length / layer_width # レイヤー上の１ポイントがカバーするオリジナル画像上のピクセル数（layer_width=19, img_width=300ならstep_x=15.78）
        # step_y = img_height / layer_height
        linx = np.linspace(0.5 * step_x, data_length - 0.5 * step_x,
                           layer_width) # img_width=300, layer_width=19 なら0-300の区間を19に分けた時のピクセル中心位置の数列（7.89, 23,68, ..., 292.105）
        # liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)

        # centers_x = np.array(linx)
        # centers_x, centers_y = np.meshgrid(linx, liny)
        # centers_x = centers_x.reshape(-1, 1)
        # centers_y = centers_y.reshape(-1, 1)
        # define xmin, ymin, xmax, ymax of prior boxes
        num_priors_ = len(self.aspect_ratios)
        # prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = linx.reshape(-1,1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors_)) # 「1, 」が必要かどうかはよくわからない…1ならなくても結果は同じ？それとも次元が一つ増える？
        prior_boxes[:, ::2] -= box_widths
        # prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 1::2] += box_widths
        # prior_boxes[:, 3::4] += box_heights
        prior_boxes[:, :] /= data_length
        # prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 2)
        if self.clip: # prior_boxのxmin, ymin, xmax, ymaxは0-1でクリップしておく
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
        # define variances
        num_boxes = len(prior_boxes)
        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 2)) * self.variances[0]
        elif len(self.variances) == 2:
            variances = np.tile(self.variances, (num_boxes, 1)) # ここでvalianceを作る
        else:
            raise Exception('Must provide one or two variances.')
        prior_boxes = np.concatenate((prior_boxes, variances), axis=1) # 作ったvalianceをconcatenateする shape: (priorboxのサイズ, 2+2)

        """priorsを保存する"""
        # temp_priors = []
        # if os.path.exists('priors_test.pkl'):
        #     with open('priors_test.pkl', mode='rb') as f:
        #         temp_priors = pickle.load(f)        
        # if len(temp_priors) != 0:
        #     temp_priors = np.concatenate((temp_priors, prior_boxes), axis=0)
        # else:
        #     temp_priors = prior_boxes
        # with open('priors_test.pkl', mode='wb') as f:
        #     pickle.dump(temp_priors, f)
        """ここまで"""

        prior_boxes_tensor = K.expand_dims(K.variable(prior_boxes), 0) # バックエンドテンソルに変換（１次元追加）shape:TensorShape([Dimension(1), Dimension(54), Dimension(8)])
        if K.backend() == 'tensorflow':
            pattern = [tf.shape(x)[0], 1, 1] # patternのshapeは(none, 1, 1)的な感じ。tf.shape(x)[0]はバッチ数
            prior_boxes_tensor = tf.tile(prior_boxes_tensor, pattern) # TensorShape([Dimension(None), Dimension(54), Dimension(8)]) これはバッチ数だけタイルされた形（バッチ数はNoneで予約）
        elif K.backend() == 'theano':
            #TODO
            pass
        return prior_boxes_tensor