# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:16:48 2016

Script to plyaback white gaussian noise from the Vifa speakers


@author: tbeleyur
"""
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

FS=192 * 1000 # sampling rate
playbackdurn=3.0 # length of playback sound
numplaybacks=2
silencedurn=3
rampduration=0.5 # length of up and down ramp in seconds
amplification=10 # in dB


playbacksamples=int(playbackdurn*FS)
rampsamples=int(rampduration*FS)

raw_gaussiannoise=np.random.normal(0.0,0.05,playbacksamples)

hamming_window=np.hamming(2*rampsamples)

windowed_noise=np.copy(raw_gaussiannoise)


windowed_noise[0:rampsamples]=hamming_window[0:rampsamples]*raw_gaussiannoise[0:rampsamples]
windowed_noise[-rampsamples:playbacksamples]=hamming_window[-rampsamples:2*rampsamples]*raw_gaussiannoise[-rampsamples:playbacksamples]


silence_playback=np.zeros(silencedurn*FS)

unitplayback=np.hstack((windowed_noise,silence_playback))

completeplayback=np.tile(unitplayback,numplaybacks)*10**(amplification/20.0)



# now to initiate sounddevice for the simultaneousl recording and playback:
print('recording and playback intiated....')
recordedsound=sd.playrec(completeplayback,samplerate=FS,channels=1,blocking=True)
print('recording and playback stopped....')
#
#deviceid=40 # 40 for the Fireface ASIO USB














if __name__=='__main__':

    plt.figure(1)
    fig=plt.subplot(111)
    fig.set_xlim(xmin=0,xmax=unitplayback.size)
    fig.set_ylim(ymin=-1,ymax=1)
    plt.plot(unitplayback)


#    plt.figure(2)
#    fig=plt.subplot(111)
#    fig.set_xlim(xmin=0,xmax=completeplayback.size)
#    fig.set_ylim(ymin=-1,ymax=1)
#    plt.plot(completeplayback)

    plt.figure(3)
    fig=plt.subplot(111)
    fig.set_xlim(xmin=0,xmax=completeplayback.size)
    fig.set_ylim(ymin=-1,ymax=1)
    plt.plot(recordedsound)