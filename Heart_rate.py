from __future__ import print_function, division, unicode_literals

import wave
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.stats import linregress

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def percentChange(startPoint,currentPoint):
    try:
        x = ((float(currentPoint)-startPoint)/abs(startPoint))*100.00
        if x == 0.0:
            return 0.000000001
        else:
            return x
    except:
        return 0.0001


#def patternRec(x, window_width):


def cal_slope(x):
    a = np.array(x)

    base = min(a)
    res = []

    for i in  a:
        lin = linregress(i, base)
        res.append(lin[0])
    return res    


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except (ValueError, msg):
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def get_peaks(x, win):
    ind = win
    peaks_y = []
    peaks_x = []
    flag = False

    while(ind < len(x)):
        flag = False
        if(ind + win < len(x)):
            for i in range(1, win+1):
                if(x[ind] > x[ind - i] and x[ind] > x[ind+i]):
                    j=0
                else:
                    flag = True
                    break
        else:
            for i in range(1, len(x) - ind):
                if(x[ind] > x[ind - i] and x[ind] > x[ind+i]):
                    j=0
                else:
                    flag = True
                    break
        if(flag == True):
            ind+=1
        else:
            #print("elseee")
            peaks_x.append(ind)
            peaks_y.append(x[ind])
            ind+=win
    return peaks_x, peaks_y

def smooth(x, win, order):
    a = np.array(x)

    res = []
    res = savitzky_golay(a, win*t + 1, order)

    return res

def smooth1(a):
    res = []
    for i in range(3,len(a)-2):
        res[i] = float((a[i-2] + a[i-1] + a[i] + a[i+1] + a[i+2])/5)

    return res

def smooth2(x, window_width):
    
    data = np.array(x)
    
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

    return ma_vec


def removeOutliers(x, c):
    a = np.array(x)

    mean = np.mean(a)
    sd = np.std(a)

    resultList = []
    for i in a.tolist():
        if i >= mean - c*sd and i<= mean + c*sd:
            resultList.append(i)
        else:
            resultList.append(0)
    return resultList

def find_fft(da):
    result = []
    temp_fft = []
    temp_inverse = []
    mono = np.zeros(t)
    ind = 0
    while ind < len(da):
        #print('Index is {}'.format(ind))
        
        
        
        if(par[0] == 1):
            
            mono[0:t//2] = mono[t//2:t]
            mono[t//2:t] = da[ind:ind+(t//2)]
            temp_fft = np.fft.rfft(mono)
            #result_fft = temp_fft[0:t/2-1]
            for i in range(0, len(temp_fft)):
                if(i<lowpass and i>highpass):
                    temp_fft[i] = 0
            temp_inverse = np.fft.irfft(temp_fft)
            result[ind: ind+(t//2)] = temp_inverse[0: t//2]
            ind+= (t//2)
    return result

def av_points(res, win):
    l = len(res)
    result = []
    win_p = win * t

    it = math.ceil((l/t)/win)
    print(it)
    rem = n - (it*win)

    buf = []
    for i in range(0,int(it)):
        buf = res[int(win_p * i) : int(win_p * (i+1))]
        result.append(mean(buf))
    
    #result.append(mean(res[-rem : -1]))
    return result



if __name__ == "__main__":

    
    wr = wave.open('Audio.wav', 'r')


    par = list(wr.getparams()) # Get the parameters from the input.
    print(par)

    n = wr.getnframes()

    t = wr.getframerate()

    time = int(n/t)
    
    lowpass = 1 * (t/(n/2)) # Remove lower frequencies.
    highpass = 10 * (t/(n/2))# Remove higher frequencies.

    da = np.fromstring(wr.readframes(time*t), dtype=np.int16)

    #print(n)




    win = 0.025

    result = find_fft(da)
    result = removeOutliers(result, 4)

    #result1_smooth = smooth(result1, 0.01*t, 3)
    result1_smooth = smooth2(result, win * t)
    result1 = av_points(result1_smooth, win)
    result1_peaks_x, result1_peaks_y = get_peaks(result1, 10)
    print(len(result1_peaks_x),len( result1_peaks_y))

    #ww.writeframes(np.array(result).tostring())
    #ww.close()
    
    #plt.plot(np.arange(0,time,1/(t)), da)
    #plt.show()
    #plt.plot(np.arange(0,time,1/(t)), result)
    plt.figure(1)
   
    plt.plot(np.arange(0,time,win), result, '-gD',markevery = result1_peaks_x)

    plt.figure(2)
    try:
        plt.plot(np.arange(0, time-win, 1/t), result1_smooth[:])
    except:
        plt.plot(np.arange(0, time-win, 1/t), result1_smooth[:-1])
    #plt.scatter(list(peaks), result[list(peaks)], 'r*')
    plt.show()

    wr.close()
    #ww.close()
