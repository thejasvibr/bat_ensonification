# -*- coding: utf-8 -*-
"""
Script that plays multiple linear sweeps in succession :

Created on Wed Feb 15 15:10:54 2017

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

RECORDING_ANGLE = 90
PLAYBACK_DISTANCE =  1 # IN METRES
GAIN = [30]

playback_freqs = '20-96KHzSWEEP'
playback_type = 'TARGET_STRENGTH'
# CHECK THE FILENAME BEFORE DOING ANYTHING AT ALL !!
time_stamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
fname = '%s_%sdeg_WITH_BAT_3msSWEEP_playback_%smetre_single_tones_%sHz_%sgain_%s.WAV'%(playback_type,RECORDING_ANGLE,PLAYBACK_DISTANCE,playback_freqs,GAIN,time_stamp)

# location to where the generated data is saved to as numpy arrays and wav file
target_folder = 'C:\\Users\\tbeleyur\\Desktop\\ensonification_data\\2017_02_17\\bat_linear_sweep_recordings\\3ms_linear_sweeps\\'

# location to get the linear sweep array from:
sweep_location_folder = 'C:\\Users\\tbeleyur\\Desktop\\ensonification_data\\2017_02_15\\linear_sweep_playback_file\\3ms_20-96KHz_sweep\\'
sweep_wcIR_array = 'cIR_conv_signal_2017-02-15_15-05.npy'

fs = 192000 # sampling rate
in_ch = [2,9]
out_ch = [2,1]

one_sweep = np.load(sweep_location_folder+sweep_wcIR_array)

# get the Fireface USB index number :
device_list = sd.query_devices()
tgt_dev_name = 'ASIO Fireface USB'
tgt_dev_bool = [tgt_dev_name in each_device['name'] for each_device in device_list]
tgt_ind = int(np.argmax(np.array(tgt_dev_bool)))

repeat_sweeps =np.tile(one_sweep,5)

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


#sectionrms = 20*np.log10([np.std(rec_sines[silence_samples:-silence_samples,1]) ] )
#print('dB rms is : ', sectionrms)