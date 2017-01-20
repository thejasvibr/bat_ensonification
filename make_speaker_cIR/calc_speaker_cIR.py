# -*- coding: utf-8 -*-
"""
Script which calculates the cIR for the speaker
to get a flat frequency response

Created on Tue Jan 17 13:55:20 2017

User records sound from a flat-response microphone
like a GRAS 1/4th inch and as output the compensatory
IR of the particular speaker is generated

based on cIR script written by Holger R Goerlitz

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

def gen_gaussian_noise(num_samples,mean,sdev):
    # generates gaussian distribution of sample values

    gaussian_noise = np.random.normal(mean,sdev,num_samples)
    return(gaussian_noise)

def filter_signal(input_signal,order,freq_fraction,filter_type):
    # lowpass,highpass or bandpass es a signal with a butterworth of
    # given order
    b,a = signal.butter(order,freq_fraction,btype=filter_type)
    filtered_signal = signal.lfilter(b,a,input_signal)
    return(filtered_signal)

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
    cross_cor = signal.correlate( input_sig_flat, rec_sig_flat, 'same')

    # get the index of maximum correlation
    max_corr = input_sig_flat.size/2 - np.argmax(abs(cross_cor))

    # choose the cross corr along a particular window size
    impulse_resp = cross_cor[max_corr-ir_length:max_corr+ir_length]
    impulse_resp_fft = spyfft.rfft(impulse_resp)
    ir_freqdBs = 20*np.log10(abs(impulse_resp_fft))
    ir_freqs = np.linspace(0,FS/2,ir_length*2)


    return( [impulse_resp,impulse_resp_fft, ir_freqs, ir_freqdBs])



def calc_cIR(impulse_resp,ir_length,lp_fraction):
    # create a Dirac pulse (which has aLL frequencies)
    dirac_pulse = np.zeros(ir_length)
    dirac_pulse[ir_length/2 -1] = 1

    b,a = signal.butter(16,lp_fraction,btype='highpass')

    # filter the frequencies for the dirac pulse and the recorded IR:
    dirac_pulse_filtered = signal.lfilter(b,a,dirac_pulse)
    impulse_response_filtered = signal.lfilter(b,a,impulse_resp)

    # calculate the fft's of both filtered signals :
    filt_dpulse_fft = spyfft.fft(dirac_pulse_filtered)
    filt_iresp_fft = spyfft.fft(impulse_response_filtered)

    # now divide the all frequency signal w the some-frequency signal :
    # to get the cIR :
    cIR_fft = filt_dpulse_fft / filt_iresp_fft

    # calculate the iFFT to get a compensatory IR filter :
    cIR = spyfft.ifft(cIR_fft).real # here HRG shifts array circularly ..why ?

    cIR_final = np.roll(cIR,ir_length/2)

    return(cIR_final)

flatten = np.ndarray.flatten # one line function assignment

def oned_fft_interp(new_freqs,fft_freqs,fft_var,interp_type='linear'):

    interp_power_spectrum= interp1d(flatten(fft_freqs), flatten( 20*np.log10(abs(fft_var)) ),kind = interp_type )
    return(interp_power_spectrum(new_freqs))





durn_pbk = 0.5
FS = 192000
numramp_samples = 0.1*FS
mic_speaker_dist = 3 # in meters
vsound = 330 # in meters/sec
delay_time = mic_speaker_dist/vsound
ir_length = 2048

total_num_samples = int(durn_pbk*FS)

gaussian_noise = gen_gaussian_noise(total_num_samples,0,0.1)
filt_gaussian_noise = filter_signal(gaussian_noise, 16,0.2,'highpass')

# trigger spike to get the playback delay :
trigger_sig = np.zeros(total_num_samples)
trigger_sig [0] = 0.8



pbk_sig =  add_ramps( numramp_samples ,filt_gaussian_noise )

final_pbk = np.column_stack((trigger_sig,pbk_sig ))

print('raw sound being played now...')
rec_sound = sd.playrec(final_pbk,FS,input_mapping=[2,11],output_mapping=[2,1] ,dtype='float',device = 40)
sd.wait()

print('signal processing happening now...')

delay_samples = int(delay_time *FS)

intfc_pbk_delay = np.argmax(rec_sound[:,0]) # audio interface playback delay

irparams = get_impulse_response(pbk_sig,rec_sound[intfc_pbk_delay:,1],ir_length,FS,delay_samples)

print('impulse and frequency response being calculated now...')

cir = calc_cIR(irparams[0],ir_length*2,0.1)

print('signal being convolved with cIR now ')
# SHOULD I CONVOLVE WITH THE 'SAME' option ....?
corrected_sig = np.convolve(pbk_sig,np.flipud(cir))

print('corrected_sound being played now...')
amp_dB = -( 20*np.log10(np.std(corrected_sig)) - 20*np.log10(np.std(pbk_sig)) )   # in dB

amp_factor = 10**(amp_dB/20.0)

rec_corrected_sound = sd.playrec(amp_factor*corrected_sig,FS,input_mapping=[11],output_mapping=[1] ,dtype='float',device = 40)
sd.wait()

plt.figure(1)
plt.plot(cir)

# smoothing over a Kilohertz range
min_plot_freq = 20000
max_plot_freq = int(FS/2)
freq_range = max_plot_freq - min_plot_freq
smoothing_freqs = np.linspace(min_plot_freq,max_plot_freq,freq_range/1000)


plt.figure(3)

plt.subplot(411)
plt.plot(rec_sound[intfc_pbk_delay:,1])
plt.title('original recorded signal')

plt.subplot(412)
orig_fft = spyfft.rfft(rec_sound[delay_samples:])
num_freqs= np.linspace(0,FS/2,orig_fft.size)
plt.plot(num_freqs,20*np.log10(abs(orig_fft)))
plt.title('FFT original recorded sound')
plt.ylim(-80,0)

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
plt.ylim(-80,0)

sm_fft= oned_fft_interp(smoothing_freqs,num_freqs_crct,crct_sig_fft)
plt.plot(smoothing_freqs,sm_fft)

plt.figure(4)
sm_fft= oned_fft_interp(smoothing_freqs,num_freqs_crct,crct_sig_fft)
plt.plot(smoothing_freqs,sm_fft-np.max(sm_fft),'r')
sm_fft_orig= oned_fft_interp(smoothing_freqs,num_freqs,orig_fft)
plt.plot(smoothing_freqs,sm_fft_orig-np.max(sm_fft_orig),'g')


#fft_res = spyfft.rfft(ccor)
#plt.plot(np.linspace(0,FS/2,2048),20*np.log10(abs(fft_res)))


