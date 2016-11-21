# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:16:48 2016

Script to plyaback white gaussian noise from the Vifa speakers


@author: tbeleyur
"""
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io.wavfile as wavfile
import scipy.io
import os
import matplotlib.pyplot as plt

FS=192 *1000 # sampling rate
playbackdurn=5.0 # length of playback sound
silencedurn=2
numplaybacks=7

rampduration=0.1 # length of up and down ramp in seconds
amplification=0 # in dB

speaker_cIR_address='C:\\Users\\tbeleyur\\Google Drive\\Holger Goerlitz- IMPRS\\PHD_2015\\projects and analyses\\bat_ensonification\\HRTF and conspecific sensing_project\\Vifa_speaker_directionality\\calibration_Vifa_2016_07_21'
speaker_cIR_filename='compIR_2016_07_21_compIR_1-90KHz.mat'
cIR_matfile=os.path.join(speaker_cIR_address,speaker_cIR_filename)

raw_speaker_cIR=np.transpose(scipy.io.loadmat(cIR_matfile)['irc'] )
speaker_cIR=np.ndarray.flatten(raw_speaker_cIR)

highpass_frequency=1*1000

playbacksamples=int(playbackdurn*FS)
rampsamples=int(rampduration*FS)
silencesamples=int(silencedurn*FS)

### create the gaussian noise to be played back

raw_gaussiannoise=np.random.normal(0.0,0.05,playbacksamples)


# compensate the Impulse Response of the Vifa speaker - to obtain a uniform
# noise spectrum :

speaker_comp_noise=compensateIR(raw_gaussiannoise,speaker_cIR) # DO I NEED TO DO A 'SUM' THING HERE  - AS SHOWN IN THE SCIPY DOCS PAGE ?


# highpass filter the signal

a,b=scipy.signal.butter(8,float(highpass_frequency)/(FS/2),btype='high')

hp_filt_gaussiannoise=scipy.signal.lfilter(a,b,speaker_comp_noise)


# perform windowing at front and back to have a slow ramp up and ramp down
hamming_window=np.hamming(2*rampsamples)

windowed_noise=np.copy(hp_filt_gaussiannoise)

windowed_noise[0:rampsamples]=hamming_window[0:rampsamples]*raw_gaussiannoise[0:rampsamples]
windowed_noise[-rampsamples:playbacksamples]=hamming_window[-rampsamples:2*rampsamples]*raw_gaussiannoise[-rampsamples:playbacksamples]

# create the sync-channel playback

silence_playback=np.zeros(silencedurn*FS)

unitplayback=np.hstack((windowed_noise,silence_playback))

startandend_silence=np.zeros(silencesamples)


# add some silence to the noise playback
finalplayback=np.hstack((startandend_silence,np.tile(unitplayback,numplaybacks)*10**(amplification/20.0),startandend_silence))

# create the syncsignal
syncsignal=np.zeros(finalplayback.size)

syncsignal[startandend_silence.size+1]=1 # indicate the start of signal

DeviceIndex=40 # for Fireface ASIO USB


# stack the two output signal  as 2 columns
playbackarray=np.column_stack((finalplayback,syncsignal,finalplayback))


# initiate sounddevice for the simultaneous recording and playback:
print('\n recording and playback intiated....')
recordedsound=sd.playrec(playbackarray,samplerate=FS,output_mapping=[1,2,3],input_mapping=[2,9,3],blocking=True,device=DeviceIndex)
print('\n recording and playback stopped....')



def convert2wav(recordedarray,fullpathnames):
        for column in range(recordedarray.shape[1]):
            filewriter(recordedarray[:,column],fullpathnames[column])

def compensateIR(inputsignal,cIR):
    try:
        outputsignal=scipy.signal.convolve(inputsignal,cIR,mode='same')

    except:
        print('there is an error in the inputsignal or cIR')

    return(outputsignal)









if __name__=='__main__':

#
#    plt.figure(1)
#    fig=plt.subplot(111)
#    fig.set_xlim(xmin=0,xmax=unitplayback.size)
#    fig.set_ylim(ymin=-1,ymax=1)
#    print('\n plots are being plotted')
#    plt.title('single noise playback unit')
#    plt.plot(unitplayback)


#    plt.figure(2)
#    fig=plt.subplot(111)
#    fig.set_xlim(xmin=0,xmax=completeplayback.size)
#    fig.set_ylim(ymin=-1,ymax=1)
#    plt.plot(completeplayback)
    plt.rcParams['agg.path.chunksize'] = 100000

    plt.figure(3)
    fig=plt.subplot(111)
    fig.set_xlim(xmin=0,xmax=finalplayback.size/float(FS))
    fig.set_ylim(ymin=-1,ymax=1)
    plt.plot(np.linspace(0,finalplayback.size/float(FS),finalplayback.size) ,recordedsound)
    plt.xlabel('time (seconds)')
    plt.ylabel('recorded signal')
    plt.title('All recorded signals - amplitude plot')


    plt.figure(5)
    f,t,Sxx=scipy.signal.spectrogram(hp_filt_gaussiannoise,FS)
    plt.title('High pass filtered signal at %d Hz'%(highpass_frequency))
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    print('plots are ready')

    targetdir='C:\\Users\\tbeleyur\\Documents\\speaker_directionality_measurements'

    filenames=['\\sync_signal.wav','\\Sanken_11.wav','\\internal_record.wav']

    fullpaths=map(lambda x: targetdir+x,filenames)

    filewriter=lambda np_array,file_name: wavfile.write(file_name,FS,np_array)



    convert2wav(recordedsound,fullpaths)

    readthisfile=targetdir+'\\mic_30dB_amplifier_-24dB'+filenames[1]
    readfs,a=wavfile.read(readthisfile)


    plt.figure(6)
    f1,t1,sxx1=scipy.signal.spectrogram(a)
    plt.pcolormesh(t1,f1,sxx1)
    plt.colorbar()



    plt.figure(7)
    subfig=plt.subplot(211)
    fraw,traw,sraw=scipy.signal.spectrogram(raw_gaussiannoise)
    plt.pcolormesh(traw,fraw,sraw)
    plt.colorbar()
    plt.title('raw gaussian noise')

    plt.subplot(212)
    fcomp,tcomp,scomp=scipy.signal.spectrogram(speaker_comp_noise)
    plt.pcolormesh(tcomp,fcomp,scomp)
    plt.colorbar()
    plt.title('speaker compensated gaussian noise')


