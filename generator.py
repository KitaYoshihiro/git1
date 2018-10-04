import numpy as np
from peakmodel import PeakModel

class Generator(object):
    def __init__(self, batch_size, datapoints, dwelltime=1, min_peaknumber=1, max_peaknumber=10, peak_dynamicrange=3, min_peakwidth=8, max_peakwidth=200):
        self.batch_size = batch_size
        self.datapoints = datapoints
        self.dwelltime = dwelltime
        self.min_peaknumber = min_peaknumber
        self.max_peaknumber = max_peaknumber
        self.peak_dynamicrange = peak_dynamicrange
        self.min_peakwidth = min_peakwidth
        self.max_peakwidth = max_peakwidth

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
                _input, _factor = PeakModel.normalize_and_spike(_input)
                _output, _factor = PeakModel.normalize(_output, _factor)
                inputs.append(_input)
                outputs.append(_output)
            yield np.array(inputs).reshape(-1,self.datapoints), np.array(outputs).reshape(-1,self.datapoints)

