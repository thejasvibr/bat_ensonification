# -*- coding: utf-8 -*-
"""
Noise playback and file saving functions 

Created on Wed Jan 25 09:52:17 2017

@author: tbeleyur
"""

import numpy as np 
import sounddevice as sd 
import scipy.io.wavfile as wav 

def load_playback_sound(file_locn):
    '''
    loads the playback WAV file as a np.array
    '''
    
    FS , playback_sound = wav.read(file_locn)
    
    return(FS , playback_sound)
    
def load_playback_array(sound_array):
    '''
    loads the Numpy array to be played back
    '''
    playback_array = np.load(sound_array)
    
    return(playback_array)
    
def include_sync_signal(pbk_sound):
    '''
    adds the sync trigger channel to the playback sound 
    returns a 2 channel np.array with Nsamples x 2 shape
    '''
    sync_sig = np.zeros(pbk_sound.size)
    sync_sig[0] = 1 
    
    full_pbk_sig = np.column_stack((sync_sig,pbk_sound))
    
    return(full_pbk_sig)

def remove_pre_sync(rec_sound):
    '''
    outputs the recorded sound after removing samples before playback 
    was initiated 
    '''
    pbk_start = np.argmax(abs(rec_sound[:,0]))
    
    post_pbk = rec_sound[pbk_start:,1]
    
    return(post_pbk)

def save_rec_file(rec_sound,FS,tgt_file):
    '''
    saves the rec_sound into the tgt_file.
    '''
    try:
        wav.write(tgt_file,FS,rec_sound)
        print('%s saved succesfully to location'%tgt_file)
    except:
        print('file saving not succesful - please check entries')

    
    