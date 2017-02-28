# -*- coding: utf-8 -*-
"""
Set of functions which extract echoes from a repeated playback recording:

Created on Tue Feb 28 16:11:18 2017

@author: tbeleyur
"""
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize']=10000
import peakutils.peak as peak

def extract_sounds(pbk_rec,sound_indxs):
    '''
    loads a signal and returns the sounds (echo OR call) as np.arrays in a list
    input:
        pbk_rec : np.array. recording with just centrally recorded playbacks
                    or direct sweep recordings with echoes.
        sound_indxs: list with 1x2 array-like entries. list with start and end
                        sample index numbers.

    Output:
        all_sounds: list with np.arrays. Each np.array has the corresponding
                        section as defined by the start-end indices of sound_indxs.
    '''
    all_sounds = []
    for each_sound in sound_indxs:
        all_sounds.append(pbk_rec[each_sound[0]:each_sound[1]])

    return(all_sounds)

def get_rectimes(pbk_rec,pbk_samples,min_peak_dist,threshold):
    '''
    returns the time points at which direct sweep recordings and
    echoes were registered at
    '''
    abs_pbk = np.abs(pbk_rec)
    rect_window = np.ones(pbk_samples)
    pbk_sum = np.convolve(abs_pbk,rect_window,'same')
    # normalise the running sum :
    pbk_sum *= 1/np.max(np.abs(pbk_sum))

    call_echo_indxs = peak.indexes(pbk_sum,threshold,min_peak_dist)

    if np.remainder(call_echo_indxs.size,2) >0.0:
        print('the resulting indexes might have an echo or call missing!')

        return(call_echo_indxs)
    else:
        return(call_echo_indxs)

def norm_16int_to_float(pbk_rec):
    '''
    converts a 16int signed np array to a float64 np.array
    '''
    max_val = -1.0 + float(2**15)

    float_rec = pbk_rec.copy()/max_val

    return(float_rec)

start_end_points = lambda index,half_samples: [index-half_samples, index + half_samples]

gen_time = lambda X,fs : np.linspace(0,X.size/float(fs),X.size)

rms_sweep = lambda indexs,sig : np.std(sig[indexs[0]:indexs[1]])

t2sample = lambda time,fs: int(time*fs)

samp2time = lambda sample,fs : sample/float(fs)

make_pow_spectrum = lambda sig : 20*np.log10(abs(np.fft.rfft(sig)))

hp19b, hp19a = signal.butter(8,19000.0/192000,'highpass')