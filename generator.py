import numpy as np
import pickle
from peakmodel import PeakModel

class Generator(object):
    def __init__(self, batch_size, datapoints, dwelltime=1, min_peaknumber=1, max_peaknumber=10, peak_dynamicrange=3, min_peakwidth=8, max_peakwidth=200, spike_noise=True):
        self.batch_size = batch_size
        self.datapoints = datapoints
        self.dwelltime = dwelltime
        self.min_peaknumber = min_peaknumber
        self.max_peaknumber = max_peaknumber
        self.peak_dynamicrange = peak_dynamicrange
        self.min_peakwidth = min_peakwidth
        self.max_peakwidth = max_peakwidth
        self.spike_noise = spike_noise

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
                _input, _output = PeakModel.chrom(self.datapoints, dwelltime=self.dwelltime,
                                                  min_peaknumber=self.min_peaknumber,
                                                  max_peaknumber=self.max_peaknumber,
                                                  peak_dynamicrange=self.peak_dynamicrange,
                                                  min_peakwidth=self.min_peakwidth,
                                                  max_peakwidth=self.max_peakwidth)
                if self.spike_noise:
                    _input, _factor = PeakModel.normalize_and_spike(_input)
                else:
                    _input, _factor = PeakModel.normalize(_input)
                _output, _factor = PeakModel.normalize(_output, _factor)
                inputs.append(_input)
                outputs.append(_output)
            yield np.array(inputs).reshape(-1,self.datapoints), np.array(outputs).reshape(-1,self.datapoints)

if __name__ == '__main__':
    # gen = Generator(batch_size=51200, datapoints=1024, spike_noise=True)
    gen = Generator(batch_size=51200, datapoints=1024, dwelltime=1,
                    min_peaknumber=1, max_peaknumber=10,
                    peak_dynamicrange=3, min_peakwidth=8,
                    max_peakwidth=200, spike_noise=True)
    g = gen.generate(train=True)
    a = np.array(next(g))
    with open('trainsample_with_noise2.pickle', mode='wb') as f:
        pickle.dump(a, f)
    # with open('sample.pickle', mode='rb') as f:
    #     b = pickle.load(f)
    # print(b.shape)