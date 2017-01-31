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
    saves the rec_sound into the tgt_file as an int16 WAV file.
    Even if rec_sound is a float array it gets converted into a int16 format
    '''
    wav2location = lambda tgt_file,FS,array: wav.write(tgt_file,FS,array)
        
    if 'float' in str(rec_sound.dtype):
        try:
            bitrate = 16 
            max_val_int = -1 + 2**(bitrate-1 )  
            int16rec_sound = np.int16(rec_sound*max_val_int)
            print('conversion from float to int happened succesfully')
            
            wav2location(tgt_file,FS,int16rec_sound)
            print('saving succesfult to %s'%tgt_file)
            return(int16rec_sound)
                 
            
        except:
            print('unsuccessful float to in16 conversion')
        
        
        
        
    else:
        print('integer arrays not supported - please save directly with scipy.io.wav')
    

   
def find_device_index(tgt_dev_name):
    '''
    searches for device_name in 
    the devices and detects the device index number
    Input: string
    Output: integer
    '''
    try:
        device_list = sd.query_devices()
        tgt_dev_bool = [tgt_dev_name in each_device['name'] for each_device in device_list]
        tgt_ind = int(np.argmax(np.array(tgt_dev_bool)))
        
        return(tgt_ind)  
    except:
        print('error in getting the device index number')
        