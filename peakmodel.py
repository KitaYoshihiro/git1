import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm

class PeakModel:    
    @classmethod
    def peak(cls, maxcps, datapoints, dwelltime, skew = 0, sigma = 3, location = 0):
        location = 0
        scale = 1
        alpha = skew
#         delta = alpha / np.sqrt(1+alpha**2)
#         uz = np.sqrt(2/np.pi) * delta
#         sigmaz = np.sqrt(1.0-uz**2.0)
#         gamma = (4-np.pi)/2 * (delta*np.sqrt(2/np.pi))**3/(1-2*delta**2/np.pi)**(3/2)
#         moa = uz - (gamma * sigmaz / 2) - (np.sign(alpha))*np.exp(-2*np.pi/np.abs(alpha))
#         mode = location + scale * moa
#         _norm_ = skewnorm.pdf(x=mode, a=alpha, loc=location, scale=scale) # 標準正規分布の高さ

        times = np.linspace(-sigma, sigma, datapoints)                
        _refpeak_ = [skewnorm.pdf(x = time, a=alpha, loc=0, scale=scale) for time in times]
        _norm_ = np.max(_refpeak_)
        maxindex = np.argmax(_refpeak_)
        maxtime = times[maxindex]
        # refpeak = np.array(_refpeak_) * maxcps / _norm_
        refpeak = np.array([skewnorm.pdf(x=time, a=alpha, loc= location - maxtime, scale=scale) * maxcps / _norm_ for time in times])
        # print('maxindex:', maxindex)
        # print('maxpos:', maxtime)
        samplepeak = np.array([np.random.poisson(peak * dwelltime / 1000) * 1000 / dwelltime for peak in refpeak])
        #return times, refpeak, samplepeak    
        return refpeak
    @classmethod
    def simulate(cls, dwelltime, chrom):
        simulated = np.array([np.random.poisson(chromdata * dwelltime / 1000) * 1000 / dwelltime for chromdata in chrom])
        return simulated
    @classmethod
    def baseline(cls, level, datapoints, dwelltime):
        sample = np.array([np.random.poisson(level * dwelltime / 1000) * 1000 / dwelltime for i in np.arange(datapoints)])
        variation = np.max(sample) - np.min(sample)        
        return sample, variation
    @classmethod
    def spikenoise(cls, datapoints):
        sample = np.array([np.random.poisson(1) for i in np.arange(datapoints)])
        # print(sample)
        return sample
    @classmethod
    def zscore(cls, x, axis = None):
        xmean = np.mean(x, axis=axis, dtype='float')
        xstd  = np.std(x, axis=axis, keepdims=True)
        zscore = (x-xmean)/xstd
        return zscore
    @classmethod
    def normalize(cls, x, factor=None, axis=None):
        if factor:
            return x/factor, factor
        xmax = np.max(x, axis=axis)
        normalized = x/xmax
        return normalized, xmax
    @classmethod
    def normalize_and_spike(cls, x, noise_rate=0.03, normalization_factor=None, axis=None):
        if normalization_factor:
            return x/normalization_factor, normalization_factor
        xmax = np.max(x, axis=axis)
        normalized = x/xmax
        noise_count = int(len(normalized) * noise_rate)
        _id = np.arange(len(normalized))
        np.random.shuffle(_id)
        _id = _id[0:noise_count]
        normalized[_id] = 1        
        return normalized, xmax

    @classmethod
    def chrom(cls, datapoints, dwelltime, min_peaknumber, max_peaknumber, peak_dynamicrange, min_peakwidth, max_peakwidth):
        baselinelevel = 10**(np.random.rand() * 3)
        peaknumber = np.random.randint(min_peaknumber, max_peaknumber + 1)
        # SNRs = [3 + np.random.rand() * (10 ** (peak_dynamicrange - 1)) for i in np.arange(peaknumber)]

        base, noiselevel = PeakModel.baseline(level= baselinelevel, datapoints= datapoints, dwelltime=dwelltime)

        if noiselevel < 100:
            noiselevel = 100

        Skews = [np.random.rand() * 5 for i in np.arange(peaknumber)]
        PeakHeights = [np.random.randint(noiselevel*3, noiselevel*(3+np.random.rand()*(10**(peak_dynamicrange-1)))+1) for i in np.arange(peaknumber)]
        PeakWidths = [np.random.randint(min_peakwidth, max_peakwidth + 1) for i in np.arange(peaknumber)]
        
        Peaks = [PeakModel.peak(maxcps = PeakHeights[i], datapoints = PeakWidths[i], dwelltime = dwelltime, skew=Skews[i]) for i in np.arange(peaknumber)]
        Positions = [np.random.randint(0, datapoints) for i in np.arange(peaknumber)]

        # ゼロレベルにピークを配置してピークだけのクロマトを作成
        RefChrom = np.zeros(datapoints) + baselinelevel
        for i in np.arange(peaknumber):
            peak = Peaks[i]
            pos = Positions[i]
            width = PeakWidths[i]
            if width % 2 == 0: # 偶数
                startpos = int(pos - width/2)
                endpos = startpos + width
            else:
                startpos = int(pos - (width-1)/2)
                endpos = startpos + width
            if startpos >= 0 and endpos < datapoints:
                RefChrom[startpos:startpos+width] += peak
            else:
                if startpos < 0 and endpos < datapoints:
                    RefChrom[0:endpos] += peak[-startpos:width]
                if startpos >= 0 and endpos >= datapoints:
                    RefChrom[startpos:datapoints] = peak[0:datapoints-startpos]
        # パルスカウントシミュレーションデータを作成
        simulated = PeakModel.simulate(dwelltime, RefChrom)
        Chrom = base + simulated

        return Chrom, RefChrom

if __name__ == '__main__':
       
    CHROM, REF = PeakModel.chrom(1024, dwelltime=1, min_peaknumber=1, max_peaknumber=10, peak_dynamicrange=3, min_peakwidth=8, max_peakwidth=200)
    CHROM, factor = PeakModel.normalize_and_spike(CHROM)
    REF, factor = PeakModel.normalize(REF)
    plt.plot(CHROM)
    plt.plot(REF)
    plt.show()