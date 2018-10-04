import numpy as np
from peakmodel import PeakModel

class Generator(object):
    def __init__(self, batch_size, datapoints):
        self.batch_size = batch_size
        self.datapoints = datapoints
        
    def zscore(self, x, axis = None):
        xmean = np.mean(x, axis=axis, dtype='float')
        xstd  = np.std(x, axis=axis, keepdims=True)
        zscore = (x-xmean)/xstd
        return zscore
    def normalize(self, x, factor=None, axis=None):
        if factor:
          return x/factor, factor
        xmax = np.max(x, axis=axis)
        normalized = x/xmax
        return normalized, xmax
    def generate(self, train=True):
        """
        batchサイズ分のデータを作ってyieldし続けるgenerator
        """
        while True:
            # if train:
            #     pass
            # else:
            #     pass
            inputs = []
            outputs = []
            for _ in np.arange(self.batch_size):
                _input, _output = PeakModel.chrom(self.datapoints)
                _input, _factor = self.normalize(_input)
                _output, _factor = self.normalize(_output, _factor)
                inputs.append(_input)
                outputs.append(_output)
            yield np.array(inputs).reshape(-1,self.datapoints), np.array(outputs).reshape(-1,self.datapoints)
