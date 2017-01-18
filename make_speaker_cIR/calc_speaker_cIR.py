# -*- coding: utf-8 -*-
"""
Script which calculates the cIR for the speaker
to get a flat frequency response

Created on Tue Jan 17 13:55:20 2017

User records sound from a flat-response microphone
like a GRAS 1/4th inch and as output the compensatory
IR of the particular speaker is generated


@author: tbeleyur
"""

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.fftpack as spyfft
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import scipy.signal as signal
plt.rcParams['agg.path.chunksize'] = 10000

### generate noise signal :

def gen_white_noise(num_samples,mean,sdev):
    # generates gaussian distribution of sample values

    white_noise = np.random.normal(mean,sdev,num_samples)
    return(white_noise)

def add_ramps(half_ramp_samples,orig_signal):
    # adds up and down ramps to the original signal
    full_window = np.hamming(2*half_ramp_samples)
    up_ramp = full_window[:half_ramp_samples]
    down_ramp = full_window[-half_ramp_samples:]

    sig_w_ramp = np.copy(orig_signal)
    sig_w_ramp[:half_ramp_samples] = orig_signal[:half_ramp_samples]*up_ramp
    sig_w_ramp[-half_ramp_samples:] = orig_signal[-half_ramp_samples:]*down_ramp

    return(sig_w_ramp)

def get_freq_response(input_signal,rec_signal,ir_length,FS,exp_delaysamples):

    input_sig_flat = np.ndarray.flatten(input_signal)
    rec_sig_flat = np.ndarray.flatten(rec_signal[exp_delaysamples:])
    cross_cor = np.convolve( rec_sig_flat, input_sig_flat)

    max_corr = np.argmax(abs(cross_cor))

    # choose the cross corr along a particular window size
    impulse_resp = cross_cor[max_corr-ir_length:max_corr+ir_length]
    impulse_resp_fft = spyfft.rfft(impulse_resp)
    ir_freqdBs = 20*np.log10(abs(impulse_resp_fft))
    ir_freqs = np.linspace(0,FS/2,ir_length*2)


    return( [impulse_resp,impulse_resp_fft, ir_freqs, ir_freqdBs])

def calc_cIR(impulse_resp_fft,ir_length,lp_fraction,hp_fraction=1):
    # create a Dirac pulse (which has aLL frequencies)
    dirac_pulse = np.zeros(ir_length)
    dirac_pulse[ir_length/2] = 1

    a,b = signal.butter(8,[lp_fraction,hp_fraction],btype='bandpass')

    # filter the frequencies for the dirac pulse and the recorded IR:
    dirac_pulse_filtered = signal.lfilter(a,b,dirac_pulse)
    impulse_response_filtered = signal.lfilter(a,b,impulse_resp)

    # calculate the fft's of both signals :
    filt_dpulse_fft = spyfft.rfft(dirac_pulse_filtered)
    filt_iresp_fft = spyfft.rfft(impulse_response_filtered)

    # now divide the all frequency signal w the some-frequency signal :
    # to get the cIR :
    cIR_fft = filt_dpulse_fft / filt_iresp_fft

    # calculate the iFFT to get a compensatory IR filter :
    cIR = spyfft.irfft(cIR_fft) # here HRG shifts array circularly ..why ?

    return()






durn_pbk = 2.0
FS = 192000
numramp_samples = 0.1*FS

pbk_sig =  add_ramps( numramp_samples ,gen_white_noise(int(durn_pbk*FS),0,0.1))

rec_sound = sd.playrec(pbk_sig,FS,1,dtype='float')
sd.wait()


import statsmodels.nonparametric.smoothers_lowess as lw
def lowessmaker(data):
    smoothed_data=lw.lowess(data,range(data.shape[0]),frac=0.005)
    return(smoothed_data)

#sm_freqs = lw.lowess(np.ndarray.flatten(freq_dB),freqs,delta=100)
#plt.plot(freqs,freq_dB,'r*')
#plt.plot(sm_freqs[:,0],sm_freqs[:,1],'green')

irparams = get_freq_response(pbk_sig,rec_sound,1024,FS,0)
#fft_res = spyfft.rfft(ccor)
#plt.plot(np.linspace(0,FS/2,2048),20*np.log10(abs(fft_res)))


