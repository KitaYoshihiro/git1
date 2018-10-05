import numpy as np
from peakmodel import PeakModel
from random import shuffle
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
                inputs.append(input_data[i])
                targets.append(output_data[i])
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield tmp_inp, tmp_targets

if __name__ == '__main__':
    with open('sample.pickle', mode='rb') as f:
        tr = pickle.load(f)
    print(tr.shape)
    gen = GdriveGenerator(batch_size=4, train_data=tr, validate_data=tr)
    g = gen.generate(train=True)
    a = np.array(next(g))
    print(a.shape)
