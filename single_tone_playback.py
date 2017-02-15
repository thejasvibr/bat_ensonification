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

RECORDING_ANGLE = 180.1
PLAYBACK_DISTANCE =  1 # IN METRES
GAIN = [22.5]

playback_freqs = '26.933KHz'
# CHECK THE FILENAME BEFORE DOING ANYTHING AT ALL !!
time_stamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
fname = '%sdeg_WITHOUT_playback_%smetre_single_tones_%sHz_%sgain_%s.WAV'%(RECORDING_ANGLE,PLAYBACK_DISTANCE,playback_freqs,GAIN,time_stamp)

# location to where the generated data is saved to as numpy arrays and wav file
target_folder = 'C:\\Users\\tbeleyur\\Documents\\bat_ensonification_data\\2017_02_15\\'


pbk_durn = 0.001 # in seconds
fs = 192000 # sampling rate
in_ch = [2,9]
out_ch = [2,1]

pbk_samples = int(pbk_durn*fs)
ramp_samples = int(0.0001*fs)


# define the frequencies to be played back :
playback_freqs = 26.933*1000#np.linspace(50,50,pbk_samples)*1000.0
t = np.linspace(0,pbk_durn,pbk_samples)
sweep = np.sin(2*np.pi*playback_freqs*t)*0.5

one_sweep = sweep

ramped_sweep = ir_funcs.add_ramps(ramp_samples,one_sweep)

# silence between the singe tone playbacks
silence_samples = int(0.2*fs)

# get the Fireface USB index number :
device_list = sd.query_devices()
tgt_dev_name = 'ASIO Fireface USB'
tgt_dev_bool = [tgt_dev_name in each_device['name'] for each_device in device_list]
tgt_ind = int(np.argmax(np.array(tgt_dev_bool)))


silence_signal = np.zeros(silence_samples)

sweep_w_silence = np.hstack((silence_signal,sweep,silence_signal))

repeat_sweeps =np.tile(sweep_w_silence,5)

print('recording happening now...')

rec_sines =sd.playrec(repeat_sweeps ,input_mapping = in_ch, output_mapping = out_ch,device = tgt_ind, samplerate = fs)
sd.wait()

saved_sound_SANKEN = pbksave.save_rec_file(rec_sines[:,1],fs,target_folder+'SANKEN_'+fname)
#saved_sound_GRAS = pbksave.save_rec_file(rec_sines[:,1],fs,target_folder+'GRAS_'+fname)

plt.figure(1)
time = np.linspace(0,rec_sines[:,1].size/float(fs),rec_sines[:,1].size)
plt.plot(time,rec_sines[:,1],label='SANKEN')
plt.plot(time,rec_sines[:,0],label='playback initiated')
plt.grid(10)
#plt.plot(rec_sines[:,2],label='GRAS')
plt.legend()
#plt.ylim(-1,1)

plt.figure(2)
freq_axis  = np.linspace(0,96,rec_sines[:,1].size/2)
plt.plot(freq_axis,20*np.log10(abs(np.fft.fft(rec_sines[:,1])))[:rec_sines[:,1].size/2] )


plt.figure(3)
print('spectrogram being calculated now...')
f,t,s = signal.spectrogram(rec_sines[:,1].flatten(),fs,nperseg=100,noverlap=50)
plt.pcolormesh(t,f,s)

sectionrms = 20*np.log10([np.std(rec_sines[silence_samples:-silence_samples,1]) ] )
print('dB rms is : ', sectionrms)
#
#plt.figure(4)
#print('cross-correlation happening now...')
#rec_signalcor = np.correlate(rec_sines[:,1].flatten(),all_sines_pbk.flatten(),'same')
#cor_left = - rec_signalcor.size/192000.0/2
#cor_right = -cor_left
#line = np.linspace(cor_left,cor_right,rec_signalcor.size)
#plt.plot(line,rec_signalcor,'r-')
#plt.title('autocorrelation of recorded signal')



