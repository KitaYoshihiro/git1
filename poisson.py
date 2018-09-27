import numpy as np
import matplotlib as plot
from numpy.random import *

x = 100 # msec
y = 100 # cps
lam = y * x / 1000 # 2 count per measurement
print('lam:', lam, 'count per measurement in average')
print('avg(set):', y)
chromdata = [poisson(lam) for i in np.arange(100000)]
#chromdata = np.reshape(chromdata, (-1,25))
min = np.min(chromdata) / x * 1000
max = np.max(chromdata) / x * 1000
avg = np.average(chromdata) / x * 1000
median = np.median(chromdata) / x * 1000
print('avg(meas):', avg, ' median:', median, ' min:', min, ' max:', max)
print(chromdata[1:100])