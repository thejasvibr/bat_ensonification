# -*- coding: utf-8 -*-
"""
Script that plays back single frequency sounds in succesion
This series of playbacks is used to check if the room is interfering
with expected sound transmission patterns in any way.

Created on Tue Feb 07 15:47:45 2017

@author: tbeleyur
"""
import sys,os
sys.path.append(os.path.realpath('make_speaker_cIR'))
import numpy as np
import sounddevice as sd
import calc_cIR as ir_funcs
import matplotlib.pyplot as plt
import scipy.signal as signal
plt.rcParams['agg.path.chunksize'] = 100000

import playback_saving_funcs as pbksave
import datetime as dt

def make_sinusoid(durn,number_samples,freq):
    t = np.linspace(0,durn,number_samples)
    sine_wave = np.sin(2*np.pi*freq*t) * 0.5

    return(sine_wave)

PLAYBACK_ANGLE = 0
PLAYBACK_DISTANCE =  1 # IN METRES
GAIN = [30,30]

# define the frequencies to be played back :
playback_freqs = np.array([50]) *10**3

# CHECK THE FILENAME BEFORE DOING ANYTHING AT ALL !!
time_stamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
fname = '%sdeg_playback_%smetre_single_tones_%sHz_%sgain_%s.WAV'%(PLAYBACK_ANGLE,PLAYBACK_DISTANCE,playback_freqs,GAIN,time_stamp)

# location to where the generated data is saved to as numpy arrays and wav file
target_folder = 'C:\\Users\\tbeleyur\\Documents\\bat_ensonification_data\\2017_02_13_cylinder\\'


pbk_durn = 0.5 # in seconds
fs = 192000 # sampling rate
in_ch = [2,9,10]
out_ch = [2,1]

pbk_samples = int(pbk_durn*fs)
ramp_samples = int(0.1*fs)

# silence between the singe tone playbacks
silence_samples = int(0.2*fs)

# get the Fireface USB index number :
device_list = sd.query_devices()
tgt_dev_name = 'ASIO Fireface USB'
tgt_dev_bool = [tgt_dev_name in each_device['name'] for each_device in device_list]
tgt_ind = int(np.argmax(np.array(tgt_dev_bool)))


sine_waves = [make_sinusoid(pbk_durn,pbk_samples,each_freq) for each_freq in playback_freqs]

ramped_sine_waves = [ ir_funcs.add_ramps(ramp_samples,each_sinewave) for each_sinewave in sine_waves]

silence_signal = np.zeros(silence_samples)

sines_w_silences = [np.hstack((silence_signal,each_sine,silence_signal)) for each_sine in ramped_sine_waves ]

all_sines_pbk = np.concatenate(sines_w_silences).ravel()

print('recording happening now...')

rec_sines =sd.playrec(all_sines_pbk ,input_mapping = in_ch, output_mapping = out_ch,device = tgt_ind, samplerate = fs)
sd.wait()

#saved_sound_SANKEN = pbksave.save_rec_file(rec_sines[:,1],fs,target_folder+'SANKEN_'+fname)
saved_sound_GRAS = pbksave.save_rec_file(rec_sines[:,1],fs,target_folder+'GRAS_'+fname)

plt.figure(1)
plt.plot(rec_sines[:,1],label='SANKEN')
plt.plot(rec_sines[:,2],label='GRAS')
plt.legend()
plt.ylim(-1,1)

plt.figure(2)
freq_axis  = np.linspace(0,96,rec_sines[:,1].size/2)
plt.plot(freq_axis,20*np.log10(abs(np.fft.fft(rec_sines[:,1])))[:rec_sines[:,1].size/2] )


plt.figure(3)
print('spectrogram being calculated now...')
f,t,s = signal.spectrogram(rec_sines[:,1].flatten(),fs)
plt.pcolormesh(t,f,s)

sectionrms = 20*np.log10([ np.std(rec_sines[silence_samples:-silence_samples,0]) ,np.std(rec_sines[silence_samples:-silence_samples,1]) ] )
print('dB rms is : ', sectionrms)
#
#plt.figure(4)
#print('cross-correlation happening now...')
#rec_signalcor = np.correlate(rec_sines[:,1].flatten(),rec_sines[:,1].flatten(),'same')
#plt.plot(rec_signalcor,'r-')
#plt.title('autocorrelation of recorded signal - channel 10')



