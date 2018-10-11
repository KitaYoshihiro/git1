import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm
from numba.decorators import jit

@jit
def peak(maxcps, datapoints, dwelltime, skew = 0, sigma = 3, location = 0):
  location = 0
  scale = 1
  alpha = skew
  times = np.linspace(-sigma, sigma, datapoints)   
  _refpeak_ = [skewnorm.pdf(x = time, a=alpha, loc=0, scale=scale) for time in times]
  _norm_ = np.max(_refpeak_)
  maxindex = np.argmax(_refpeak_)
  maxtime = times[maxindex]
  refpeak = np.array([skewnorm.pdf(x=time, a=alpha, loc= location - maxtime, scale=scale) * maxcps / _norm_ for time in times])
  return refpeak

@jit
def simulate(dwelltime, chrom):
  simulated = np.array([np.random.poisson(chromdata * dwelltime / 1000) * 1000 / dwelltime for chromdata in chrom])
  return simulated

@jit
def baseline(level, datapoints, dwelltime):
  sample = np.array([np.random.poisson(level * dwelltime / 1000) * 1000 / dwelltime for i in np.arange(datapoints)])
  variation = np.max(sample) - np.min(sample)        
  return sample, variation

@jit
def spikenoise(datapoints):
  sample = np.array([np.random.poisson(1) for i in np.arange(datapoints)])
  return sample

@jit
def zscore(x, axis = None):
  xmean = np.mean(x, axis=axis, dtype='float')
  xstd  = np.std(x, axis=axis, keepdims=True)
  zscore = (x-xmean)/xstd
  return zscore

@jit
def normalize(x, factor=None, axis=None):
  if factor:
    return x/factor, factor
  xmax = np.max(x, axis=axis)
  if xmax == 0:
    return x, 1
  normalized = x/xmax
  return normalized, xmax

@jit
def normalize_and_spike(x, noise_rate=0.03, normalization_factor=None, axis=None):
  if normalization_factor:
      return x/normalization_factor, normalization_factor
  xmax = np.max(x, axis=axis)
  if xmax == 0:
      normalized = x
      xmax = 1
  else:
    normalized = x/xmax
  noise_count = int(len(normalized) * noise_rate)
  _id = np.arange(len(normalized))
  np.random.shuffle(_id)
  _id = _id[0:noise_count]
  normalized[_id] = 1        
  return normalized, xmax
def chrom(datapoints, dwelltime, min_peaknumber, max_peaknumber, peak_dynamicrange, min_peakwidth, max_peakwidth):

  baselinelevel = 10**(np.random.rand() * 3)
  peaknumber = np.random.randint(min_peaknumber, max_peaknumber + 1)

  base, noiselevel = baseline(level= baselinelevel, datapoints= datapoints, dwelltime=dwelltime)

  if noiselevel < 100:
    noiselevel = 100

  Skews = [np.random.rand() * 5 for i in np.arange(peaknumber)]
  PeakHeights = [np.random.randint(noiselevel*3, noiselevel*(3+np.random.rand()*(10**(peak_dynamicrange-1)))+1) for i in np.arange(peaknumber)]
  PeakWidths = [np.random.randint(min_peakwidth, max_peakwidth + 1) for i in np.arange(peaknumber)]

  Peaks = [peak(maxcps = PeakHeights[i], datapoints = PeakWidths[i], dwelltime = dwelltime, skew=Skews[i]) for i in np.arange(peaknumber)]
  Positions = [np.random.randint(0, datapoints) for i in np.arange(peaknumber)]

  # ゼロレベルにピークを配置してピークだけのクロマトを作成
  RefChrom = np.zeros(datapoints) + baselinelevel
  for i in np.arange(peaknumber):
    pk = Peaks[i]
    pos = Positions[i]
    width = PeakWidths[i]
    if width % 2 == 0: # 偶数
      startpos = int(pos - width/2)
      endpos = startpos + width
    else:
      startpos = int(pos - (width-1)/2)
      endpos = startpos + width
    if startpos >= 0 and endpos < datapoints:
      RefChrom[startpos:startpos+width] += pk
    else:
      if startpos < 0 and endpos < datapoints:
        RefChrom[0:endpos] += pk[-startpos:width]
      if startpos >= 0 and endpos >= datapoints:
        RefChrom[startpos:datapoints] = pk[0:datapoints-startpos]
  # パルスカウントシミュレーションデータを作成
  simulated = simulate(dwelltime, RefChrom)
  Chrom = base + simulated
  return Chrom, RefChrom

if __name__ == '__main__':       
    CHROM, REF = chrom(1024, dwelltime=1, min_peaknumber=1, max_peaknumber=10, peak_dynamicrange=3, min_peakwidth=8, max_peakwidth=200)
    CHROM, factor = normalize_and_spike(CHROM)
    REF, factor = normalize(REF)
    plt.plot(CHROM)
    plt.plot(REF)
    plt.show()
