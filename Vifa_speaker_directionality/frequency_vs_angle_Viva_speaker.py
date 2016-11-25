# -*- coding: utf-8 -*-
"""

plots the frequency vs directionality for speaker output at various angles

Created on Fri Nov 25 15:02:27 2016

@author: tbeleyur
"""
import noise_analyses as nsfuncs
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack


plt.rcParams['agg.path.chunksize'] = 10000

# now testing it with a real file ! :

FS=192000
targetdir='C:\\Users\\tbeleyur\\Documents\\speaker_directionality_measurements\\23_Nov_recordings_w)speakercIR_GRASmic\\round_1\\'
internal_name='internal_record.wav'
micrec_name='GRAS_MICROPHONE.wav'
syncrec_name='sync_signal.wav'
filenames=[internal_name,micrec_name,syncrec_name]

rec_bits=16 # the number of bits the wav file has
maxval=2**rec_bits -1


playbacksamples=int(np.around(FS*3.0))
silencesamples=int(np.around(FS*1.5))
numplaybacks=19


fullpaths=map(lambda x: targetdir+x,filenames)
recsound=map(lambda filename: wavfile.read(filename)[1],fullpaths)



syncindex=nsfuncs.findsyncpulse(recsound[2])



internalsig=nsfuncs.signalpostpulse(recsound[0],syncindex)
recordedsig=nsfuncs.signalpostpulse(recsound[1],syncindex)

hp_recordedsig=nsfuncs.highpassfilter(recordedsig,1000,FS)

recordingdelay=nsfuncs.findrecordingdelay(hp_recordedsig,internalsig,1000,0.375,330,FS)

recplaybacks=nsfuncs.extractplaybacks(hp_recordedsig,playbacksamples,silencesamples,recordingdelay,numplaybacks)

norm_recplaybacks=recplaybacks/maxval
# now let's apply an fft on each recorded signal :

recordedffts=np.apply_along_axis( scipy.fftpack.rfft ,0,norm_recplaybacks,1024)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

smoothfft=np.apply_along_axis(smooth,1,20*np.log10(abs(recordedffts)),150)

plt.plot(smoothfft)



