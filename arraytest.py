""" arraytest.py
"""

import numpy as np
from numpy.random import randint

print(max(0, 2047-1023))
print(min(1024 +1 ,2047 + 1))

print(max(0, 0-1023))
print(min(1024 + 1,0 +1 ))


index = randint(max(0, 2047-1023), min(1024, 2047))
print(index)
