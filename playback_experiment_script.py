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
playback_angle = 0
rec_type ='CENTRE_PVCcylinder_at_1m_' # with_ or without_ bat
tgt_folder = 'C://Users//tbeleyur//Desktop//ensonification_data//2017_02_10//both_mics//'
GRASgain = 30
SANKENgain = 30


# initiate and record playback :
in_channels = [2,9,10] # [sync, microphone]
out_channels = [2,1] # [ sync , speaker ] 
FS = 192000

tgt_dev_name = 'ASIO Fireface USB'

ai_number = bat_enson.find_device_index(tgt_dev_name) # audio interface serial number 38 for PC opp Fluraum klein, and 40 for TBR laptop


# load the playback sound file :
amp_dB = 6.0

pbk_sound = bat_enson.load_playback_array(pbk_wav_locn+pbk_file)*10**(amp_dB/20.0)

composite_playback = bat_enson.include_sync_signal(pbk_sound)

rec_sound = sd.playrec(composite_playback, samplerate = FS, 
   input_mapping = in_channels, output_mapping= out_channels, device = ai_number)
sd.wait()

plt.plot(rec_sound)


# script puts the name together 
time_stamp = dt.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
file_name = 'playback_angle_%d_'%playback_angle + rec_type + time_stamp + '.WAV'
complete_file = tgt_folder + file_name

rec_post_sync_GRAS = bat_enson.remove_pre_sync(rec_sound[:,(0,2)])
rec_post_sync_SANKEN = bat_enson.remove_pre_sync(rec_sound[:,(0,1)])


complete_file_GRAS = tgt_folder+ 'GRASgain%s_'%GRASgain +file_name
complete_file_SANKEN = tgt_folder + 'SANKENgain%s_'%SANKENgain +file_name

rec_asint16 = bat_enson.save_rec_file(rec_post_sync_GRAS,FS,complete_file_GRAS)
rec_asint16 = bat_enson.save_rec_file(rec_post_sync_GRAS,FS,complete_file_SANKEN)

print( 'dB rms GRAS is:' ,20.0*np.log10(np.std(rec_post_sync_GRAS)) )
print( 'dB rms SANKEN is:' ,20.0*np.log10(np.std(rec_post_sync_SANKEN)) )

