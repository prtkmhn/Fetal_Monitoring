from __future__ import print_function, division, unicode_literals

import wave
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.stats import linregress


def smooth2(x, window_width):
    count=0
    data = np.array(x)
 
    for i in range(0,len(data),window_width):
        arr=np.array([])
        arr=data[i:(i+window_width-1)]
        #magnitude
        arr=np.absolute(arr)
        avg=np.average(arr)
        maxi=max(arr)
        ind=i+np.argmax(arr)
        diff=maxi-avg
        if(maxi>110 ):
            count=count+1
            for j in range(i,(i+window_width-1)):
                if(j!=ind):
                    data[j]=avg
    print(count)
    return data
wr = wave.open('Audio.wav', 'r')


par = list(wr.getparams()) # Get the parameters from the input.
print(par)

n = wr.getnframes()

t = wr.getframerate()

time = int(n/t)

da = np.fromstring(wr.readframes(time*t), dtype=np.int16)

#print(n)


win = 0.025
print(da)
plt.figure(1)
plt.plot(da)
#result1_smooth = smooth(result1, 0.01*t, 3)
result1_smooth = smooth2(da, 1000000)
plt.figure(2)
plt.plot(result1_smooth)
plt.show()