# -*- coding: utf-8 -*-
"""
Bat ensonification experiment - Grossflugraum 
script that initiates playbacks and saves the resulting recording
Created on Wed Jan 25 10:25:21 2017

@author: tbeleyur
"""
import scipy.io.wavfile as wav
import numpy as np 
import matplotlib.pyplot as plt 
import playback_saving_funcs as bat_enson
import sounddevice as sd 
import datetime as dt 
plt.rcParams['agg.path.chunksize'] = 100000
import sys
sys.stdout.flush()

# location of playback sound file  
pbk_wav_locn = 'C://Users//tbeleyur//Desktop//ensonification_data//2017_02_01_playback_sound//'
pbk_file = 'cIR_conv_signal_2017-02-01_12-07.npy'


# user input required :
# target file name :
playback_angle = 120
Fireface_gain = 30
rec_type = 'SANKEN_%sgain_without_at_1m_'%Fireface_gain # with_ or without_ bat
tgt_folder = 'C://Users//tbeleyur//Desktop//ensonification_data//2017_02_01//SANKEN_without//'



# initiate and record playback :
in_channels = [2,9] # [sync, microphone]
out_channels = [2,1] # [ sync , speaker ] 
FS = 192000

tgt_dev_name = 'ASIO Fireface USB'

ai_number = bat_enson.find_device_index(tgt_dev_name) # audio interface serial number 38 for PC opp Fluraum klein, and 40 for TBR laptop



# script puts the name together 
time_stamp = dt.datetime.now().strftime('%Y-%m-%d_%H_%M')
file_name = 'playback_angle_%d_'%playback_angle + rec_type + time_stamp + '.WAV'
complete_file = tgt_folder + file_name


# load the playback sound file :
amp_dB = 6.0

pbk_sound = bat_enson.load_playback_array(pbk_wav_locn+pbk_file)*10**(amp_dB/20)

composite_playback = bat_enson.include_sync_signal(pbk_sound)

rec_sound = sd.playrec(composite_playback, samplerate = FS, 
   input_mapping = in_channels, output_mapping= out_channels, device = ai_number)
sd.wait()

plt.plot(rec_sound)

rec_post_sync = bat_enson.remove_pre_sync(rec_sound)

rec_asint16 = bat_enson.save_rec_file(rec_post_sync,FS,complete_file)
print( 'dB rms is:' ,20.0*np.log10(np.std(rec_post_sync)) )


