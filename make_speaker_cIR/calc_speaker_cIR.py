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
from scipy.interpolate import interp1d
import scipy.signal as signal
import statsmodels.nonparametric.smoothers_lowess as lw
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

def get_impulse_response(input_signal,rec_signal,ir_length,FS,exp_delaysamples):

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



def calc_cIR(impulse_resp,impulse_resp_fft,ir_length,lp_fraction):
    # create a Dirac pulse (which has aLL frequencies)
    dirac_pulse = np.zeros(ir_length)
    dirac_pulse[ir_length/2] = 1

    b,a = signal.butter(4,lp_fraction,btype='highpass')

    # filter the frequencies for the dirac pulse and the recorded IR:
    dirac_pulse_filtered = signal.lfilter(b,a,dirac_pulse)
    impulse_response_filtered = signal.lfilter(b,a,impulse_resp)

    # calculate the fft's of both signals :
    filt_dpulse_fft = spyfft.rfft(dirac_pulse_filtered)
    filt_iresp_fft = spyfft.rfft(impulse_response_filtered)

    # now divide the all frequency signal w the some-frequency signal :
    # to get the cIR :
    cIR_fft = filt_dpulse_fft / filt_iresp_fft

    # calculate the iFFT to get a compensatory IR filter :
    cIR = spyfft.irfft(cIR_fft) # here HRG shifts array circularly ..why ?

    cIR_final = np.roll(cIR,ir_length/2)

    return(cIR_final)

flatten = np.ndarray.flatten

def oned_fft_interp(new_freqs,fft_freqs,fft_var,interp_type='linear'):

    interp_power_spectrum= interp1d(flatten(fft_freqs), flatten( 20*np.log10(abs(fft_var)) ),kind = interp_type )
    return(interp_power_spectrum(new_freqs))





durn_pbk = 1.5
FS = 192000
numramp_samples = 0.1*FS
mic_speaker_dist = 1.2 # in meters
vsound = 330 # in meters/sec
delay_time = mic_speaker_dist/vsound


pbk_sig =  add_ramps( numramp_samples ,gen_white_noise(int(durn_pbk*FS),0,0.1))

print('raw sound being played now...')
rec_sound = sd.playrec(pbk_sig,FS,dtype='float',output_mapping=[1],input_mapping=[12],device=40)
sd.wait()

print('signal processing happening now...')

delay_samples = int(delay_time *FS)
irparams = get_impulse_response(pbk_sig,rec_sound,1024,FS,delay_samples)

cir = calc_cIR(irparams[0],irparams[1],1024*2,0.01)

corrected_sig = np.convolve(pbk_sig,cir)

print('corrected_sound being played now...')
amp_dB = -( 20*np.log10(np.std(corrected_sig)) - 20*np.log10(np.std(pbk_sig)) )   # in dB

amp_factor = 10**(amp_dB/20.0)

rec_corrected_sound = sd.playrec(amp_factor*corrected_sig,FS,output_mapping=[1],input_mapping=[12],dtype='float',device=40)
sd.wait()
#plt.plot(cir)

smoothing_freqs = np.linspace(0,FS/2,50)


plt.figure(3)

plt.subplot(411)
plt.plot(rec_sound[delay_samples:])
plt.title('original recorded signal')

plt.subplot(412)
orig_fft = spyfft.rfft(rec_sound[delay_samples:])
num_freqs= np.linspace(0,FS/2,orig_fft.size)
plt.plot(num_freqs,20*np.log10(abs(orig_fft)))
plt.title('FFT original recorded sound')

sm_fft_orig= oned_fft_interp(smoothing_freqs,num_freqs,orig_fft)
plt.plot(smoothing_freqs,sm_fft_orig)


plt.subplot(413)
plt.plot(rec_corrected_sound[delay_samples:])
plt.title('cIR X original sound recorded sound')


plt.subplot(414)
crct_sig_fft = spyfft.rfft(rec_corrected_sound[delay_samples:])
num_freqs_crct= np.linspace(0,FS/2,crct_sig_fft.size)
plt.plot(num_freqs_crct,20*np.log10(abs(crct_sig_fft)))
plt.title('FFT: with cIR recorded sound')

sm_fft= oned_fft_interp(smoothing_freqs,num_freqs_crct,crct_sig_fft)
plt.plot(smoothing_freqs,sm_fft)

plt.figure(4)
sm_fft= oned_fft_interp(smoothing_freqs,num_freqs_crct,crct_sig_fft)
plt.plot(smoothing_freqs,sm_fft-np.max(sm_fft),'r')
sm_fft_orig= oned_fft_interp(smoothing_freqs,num_freqs,orig_fft)
plt.plot(smoothing_freqs,sm_fft_orig-np.max(sm_fft_orig),'g')


#fft_res = spyfft.rfft(ccor)
#plt.plot(np.linspace(0,FS/2,2048),20*np.log10(abs(fft_res)))


