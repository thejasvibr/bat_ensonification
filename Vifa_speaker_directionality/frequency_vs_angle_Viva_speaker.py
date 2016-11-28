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

hp_recordedsig=nsfuncs.highpassfilter(recordedsig,1000.0,FS)

recordingdelay=nsfuncs.findrecordingdelay(hp_recordedsig,internalsig,1000,0.375,330,FS)

recplaybacks=nsfuncs.extractplaybacks(hp_recordedsig,playbacksamples,silencesamples,recordingdelay,numplaybacks)

norm_recplaybacks=recplaybacks/maxval
# now let's apply an fft on each recorded signal :

recordedffts=np.apply_along_axis( scipy.fftpack.rfft ,0,norm_recplaybacks)


# SHOULD I BE MULTIPLYING BY 10 OR 20 ???????????????????????
windowsize=1000
# only using the real part  here - what about the Imaginary part ???!!

## CHECK FOR PROPER SMOOTHING OPTIONS - THIS IS ONLY A STANDBY !!
#smoothfft=np.apply_along_axis(smooth,1,20*np.log10(np.abs(recordedffts)),windowsize)
import statsmodels.nonparametric.smoothers_lowess as lw


scaletodBre1=lambda data: 20*np.log10(np.abs(data))

logfft=np.apply_along_axis(scaletodBre1,1,recordedffts)

def lowessmaker(data):
    smoothed_data=lw.lowess(data,range(data.shape[0]),frac=0.01,it=0,delta=10000)
    return(smoothed_data[:,1])



plt.figure(1)
xfreqs=np.linspace(0,96,logfft[:,1].shape[0])
for k in [0,1,2,3,4,8,12,18]:
    ang=k*5.0
    plt.plot(xfreqs,lowessmaker(logfft[:,k]),label=('%d deg'%ang ))
plt.xlabel('frequency - KHz')
plt.ylabel('dB intensity re 1')
plt.xlim(0,150)
plt.title('Frequency content of recorded noise across angles')
plt.legend()
plt.grid(10)
#plt.xlim(10,96)


plt.figure(3)
plt.subplot(211)
plt.plot(norm_recplaybacks[:,0])
plt.title('on-axis amplitude profile')
plt.ylim(-1,1)
plt.subplot(212)
plt.plot(norm_recplaybacks[:,18])
plt.title('90degrees off-axis amplitude profile')
plt.ylim(-1,1)

plt.figure(4)
plt.subplot(211)
t,f,s=scipy.signal.spectrogram(norm_recplaybacks[:,0])
plt.pcolormesh(f,t,scaletodBre1(s))
plt.colorbar()
plt.title('on-axis recorded playback')

plt.subplot(212)
t,f,s90=scipy.signal.spectrogram(norm_recplaybacks[:,8])
plt.pcolormesh(f,t,scaletodBre1(s90))
plt.colorbar()
plt.title('90 degrees away- recorded playback')

