# -*- coding: utf-8 -*-
"""
Script which compares the intensity of received
echoes from an object.
Created on Wed Feb 15 06:27:16 2017

@author: tbeleyur
"""
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize']=100000
import pandas as pd
import scipy.io.wavfile as wav
import os,sys
project_folder = 'C:\\Users\\tbeleyur\\Google Drive\\Holger Goerlitz- IMPRS\\PHD_2015\\projects and analyses\\bat_ensonification\\HRTF and conspecific sensing_project\\'
analysis_folder = project_folder+'analysis\\'
speaker_cir = project_folder + 'make_speaker_cIR'

sys.path.append(os.path.realpath(project_folder))
sys.path.append(os.path.realpath(analysis_folder))
sys.path.append(os.path.realpath(speaker_cir))
import calc_cIR as ir_funcs

def subtract_wo_object(rec_with,rec_without):
    '''
    arithmetic subtraction of the two np.arrays.
    '''

    rec_diff = rec_with - rec_without

    return(rec_diff)

def create_w_and_wo_pairs(folder_address,files_dataframe):
    '''
    extracts all recordings and creates pairs of
    np.arrays into a list.
    index  0 has recording without object
    index 1 has recording with object
    '''
    with_df = files_dataframe [ files_dataframe['type']=='with']
    without_df =files_dataframe [ files_dataframe['type']=='without']

    rec_pairs = []
    for each_row in range(with_df.shape[0]):

        fs,with_rec = wav.read(folder_address+with_df['file_name'].iloc[each_row])
        fs,without_rec = wav.read(folder_address+without_df['file_name'].iloc[each_row])

        rec_pairs.append([without_rec,with_rec])

    return(fs,with_df,without_df,rec_pairs)

def filter_pairs(rec_pairs,filter_order,frequency_fracs,filter_type):
    filtd_rec_pairs = []
    for each_pair in rec_pairs:

        filtd_with,ba = ir_funcs.filter_signal(each_pair[0],filter_order,frequency_fracs,filter_type)
        filtd_without,ba = ir_funcs.filter_signal(each_pair[1],filter_order,frequency_fracs,filter_type)
        filtd_rec_pairs.append([filtd_without,filtd_with])

    return(filtd_rec_pairs)





if __name__ == '__main__':

    data_folder = 'C:\\Users\\tbeleyur\\Documents\\bat_ensonification_data\\2017_02_14\\'
    files_csv = 'with_without_pingpong_1ms_playbacks.csv'
    recs_df = pd.read_csv(data_folder+files_csv)

    fs,w_df,wo_df,rec_pairs = create_w_and_wo_pairs(data_folder,recs_df)

    time = np.linspace(0,rec_pairs[0][0].size/float(fs),rec_pairs[0][0].size)

    hp_wo,ba = ir_funcs.filter_signal(rec_pairs[0][0],8,[0.3,0.9],'bandpass')

    filtered_pairs = filter_pairs(rec_pairs,8,[0.4],'highpass')


    plt.plot(time,filtered_pairs[0][1]-filtered_pairs[0][0])





