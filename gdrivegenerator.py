import numpy as np
from numpy.random import randint
from peakmodel import PeakModel
from random import shuffle
from random import random
import pickle

class GdriveGenerator(object):
    def __init__(self, batch_size, train_data, validate_data):
        self.batch_size = batch_size
        self.train_data = train_data
        self.validate_data = validate_data

    def generate(self, train=True):
        """
        batchサイズ分のデータを作ってyieldし続けるgenerator
        """
        while True:
            if train:
                x = np.random.permutation(len(self.train_data[0]))
                input_data = self.train_data[0][x]
                output_data = self.train_data[1][x]
            else:
                x = np.random.permutation(len(self.validate_data[0]))
                input_data = self.validate_data[0][x]
                output_data = self.validate_data[1][x]

            inputs = []
            targets = []
            for i in np.arange(len(input_data)):
                index = randint(1024)
                indata = input_data[i][index:index+1024]
                outdata = output_data[i][index:index+1024]
                if(random()<0.5):
                    indata = indata[::-1]
                    outdata = outdata[::-1]
                # indata = indata / np.max(indata)
                inputs.append(indata)
                # outdata = outdata / np.max(outdata)
                targets.append(outdata)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    epsilon = 0.0000001 # 10e-7
                    tmp_inp = np.log10(tmp_inp + epsilon) / 7 + 1
                    tmp_targets = np.log10(tmp_targets + epsilon) / 7 + 1
                    yield tmp_inp, tmp_targets

if __name__ == '__main__':
    with open('sample.pickle', mode='rb') as f:
        tr = pickle.load(f)
    print(tr.shape)
    gen = GdriveGenerator(batch_size=4, train_data=tr, validate_data=tr)
    g = gen.generate(train=True)
    a = np.array(next(g))
    print(a.shape)
