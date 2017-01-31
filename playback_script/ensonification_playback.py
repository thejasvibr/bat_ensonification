# -*- coding: utf-8 -*-
"""
creates a playback script to make a recording
for the bat ensonification experiments
Created on Mon Jan 23 15:10:54 2017
@author: tbeleyur
"""

'''
setup of the Fireface 802

Output channels
sync_pulse channel : Input #2 to Output #2
playback_channel : Output #1 to Harman Amplifier

Input channels
mic_channel : GRAS mic to Input #9
sync_record : records sync pulse
'''
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 100000


def load_playback_noise(file_locn):
    playback_noise = wav.read(file_locn)
    return(playback_noise )

def extract_post_pulse(rec_channels):
    '''
    cuts out the recorded signal from the point of actual playback start
    '''
    pbk_start = np.argmax(abs(rec_channels[:,0]))

    rec_post_pulse = rec_channels[pbk_start:,1]

    return(rec_post_pulse)


def write_rec_sounds(target_file,rate,sound_array):
    '''
    writes np arrays into single or multi-channel wav files
    '''
    try:
        wav.write(target_file,rate,sound_array)
        return('file %s has been succesfully written')
    except:
        print('issue with saving file')

def make_playback_signals(pback_signal):
    '''
    creates a two channel np array that can be loaded directly onto soundevice
    to play
    channel 1 : sync pulse
    channel 2: noise signal
    '''
    sync_signal = np.zeros(pback_signal.size)
    sync_signal [0] = 1

    playback_channels = np.column_stack((sync_signal,pback_signal))

    return(playback_channels)


# file location of the playback wav file
PLAYBACK_FILE = 'C://Users//tbeleyur//Desktop//test_wav.WAV'

# location to where the recorded sounds should be saved :
PLAYBACK_ANGLE = 0
RECORDING_POSITION = '_%d_degrees' %PLAYBACK_ANGLE

REC_TYPE = 'with' # with or without bat

TARGET_FOLDER = 'C://Users//tbeleyur//Documents//bat_ensonification_data//'
FILENAME = TARGET_FOLDER +REC_TYPE+ RECORDING_POSITION + '.WAV'


FS,pback_noise = load_playback_noise(PLAYBACK_FILE)

pbk_signals = make_playback_signals(pback_noise)

input_channels = [2,10]
output_channels = [2,1]
Fireface_id = 40 # otherwise try 42

rec_sound = sd.playrec(pbk_signals,samplerate = FS,input_mapping = input_channels
,output_mapping = output_channels, device = Fireface_id)
sd.wait()

rec_post_pulse = extract_post_pulse(rec_sound)

write_rec_sounds(FILENAME,FS,rec_post_pulse)




plt.figure(1)
plt.plot(rec_sound)
plt.title('recorded channels: sync channel(blue), noise(green)')

















