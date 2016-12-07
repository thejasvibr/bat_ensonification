# -*- coding: utf-8 -*-
"""m
Created on Fri Nov 18 10:16:48 2016

Script to plyaback white gaussian noise from the Vifa speakers
and record with the GRAS microphone

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
import time

starttime=time.time()

def convert2wav(recordedarray,fullpathnames):
        for column in range(recordedarray.shape[1]):
            filewriter(recordedarray[:,column],fullpathnames[column])



def compensateIR(inputsignal,cIR):
    try:
        outputsignal=scipy.signal.convolve(inputsignal,cIR,mode='same')
        return(outputsignal)

    except:
        print('there is an error in the inputsignal or cIR')






FS=192 *1000 # sampling rate
playbackdurn=3.0 # length of playback sound
silencedurn=1.5
numplaybacks=3

rampduration=0.1 # length of up and down ramp in seconds
amplification=6 # in dB

speaker_cIR_address='C:\\Users\\tbeleyur\\Google Drive\\Holger Goerlitz- IMPRS\\PHD_2015\\projects and analyses\\bat_ensonification\\HRTF and conspecific sensing_project\\Vifa_speaker_directionality\\peaker_cIR_included_Vifa_2016_07_21'
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

speaker_comp_noise=compensateIR(raw_gaussiannoise,speaker_cIR)
print('speaker cIR has been convolved with noise signal ')

# highpass filter the signal

a,b=scipy.signal.butter(8,float(highpass_frequency)/(FS/2),btype='high')

hp_filt_gaussiannoise=scipy.signal.lfilter(a,b,speaker_comp_noise)

print('high pass filtering complete at %d' %highpass_frequency)

# perform windowing at front and back to have a slow ramp up and ramp down
hamming_window=np.hamming(2*rampsamples)

windowed_noise=np.copy(hp_filt_gaussiannoise)

windowed_noise[0:rampsamples]=hamming_window[0:rampsamples]*raw_gaussiannoise[0:rampsamples]
windowed_noise[-rampsamples:playbacksamples]=hamming_window[-rampsamples:2*rampsamples]*raw_gaussiannoise[-rampsamples:playbacksamples]

# create the silence between the pulses - within which the experimenter shifts the speaker's direciton

silence_playback=np.zeros(silencedurn*FS)

unitplayback=np.hstack((windowed_noise,silence_playback))


# some silence before the whole playback - and after the whole playback

startandend_silence=np.zeros(silencesamples)


# add some silence to the noise playback
finalplayback=np.hstack((startandend_silence,np.tile(unitplayback,numplaybacks)*10**(amplification/20.0),startandend_silence))

# create the syncsignal
syncsignal=np.zeros(finalplayback.size)

syncsignal[startandend_silence.size+1]=1 # indicate the start of signal

DeviceIndex=40 # 40 for Fireface ASIO USB


# stack the two output signal  as 2 columns
playbackarray=np.column_stack((finalplayback,syncsignal,finalplayback))


# initiate sounddevice for the simultaneous recording and playback:
print('\n recording and playback intiated....')
recordedsound=sd.playrec(playbackarray,samplerate=FS,output_mapping=[1,2,3],input_mapping=[1,2,12],blocking=True,device=DeviceIndex,dtype='int16')
print('\n recording and playback stopped....')




if __name__=='__main__':

    plt.rcParams['agg.path.chunksize'] = 100000

    print('plots are being prepared')

    maxval=float(2**15 -1 )

    plt.figure(1)
    fig=plt.subplot(111)
    fig.set_xlim(xmin=0,xmax=finalplayback.size/float(FS))
    fig.set_ylim(ymin=-1,ymax=1)
    plt.plot(np.linspace(0,finalplayback.size/float(FS),finalplayback.size),recordedsound/maxval)
    plt.xlabel('time (seconds)')
    plt.ylabel('recorded signal')
    plt.title('All recorded signals - amplitude plot')


    plt.figure(2)
    f,t,Sxx=scipy.signal.spectrogram(hp_filt_gaussiannoise,FS)
    plt.title('High pass filtered signal at %d Hz'%(highpass_frequency))
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


    plt.figure(3)
    subfig=plt.subplot(211)
    fraw,traw,sraw=scipy.signal.spectrogram(raw_gaussiannoise)
    plt.pcolormesh(traw,fraw,20*np.log10(sraw))
    plt.colorbar()
    plt.title('raw gaussian noise')

    plt.subplot(212)
    fcomp,tcomp,scomp=scipy.signal.spectrogram(speaker_comp_noise)
    plt.pcolormesh(tcomp,fcomp,20*np.log10(scomp))
    plt.colorbar()
    plt.title('input signal to speaker w speaker cIR')

    plt.figure(4)

    normrec=(recordedsound[:,2])/maxval
    flat_recsound=np.ndarray.flatten(normrec)
    frec,trec,srec=scipy.signal.spectrogram(flat_recsound)
    plt.pcolormesh(trec,frec,10*np.log10(srec))
    plt.colorbar()
    plt.title('spectrogram of raw recorded sound - with GRAS microphone')

    print('plots are ready')

    targetdir='C:\\Users\\tbeleyur\\Documents\\speaker_directionality_measurements'

    filenames=['\\internal_record.wav','\\sync_signal.wav','\\GRAS_MICROPHONE.wav']

    fullpaths=map(lambda x: targetdir+x,filenames)

    filewriter=lambda np_array,file_name: wavfile.write(file_name,FS,np_array)


    convert2wav(recordedsound,fullpaths)

    print('the recordedsound has been written to this address: %s'%targetdir)

    print('%d playbacks and plotting took %d seconds'%(numplaybacks,time.time()-starttime))


    silencerec=normrec[0:193400]
    rmssilence=np.sqrt(np.mean(silencerec**2))

    playbackrec=normrec[194304:194304+playbacksamples]
    rmsplayback=np.sqrt(np.mean(playbackrec**2))

    print('Signal to noise rms ratio is : %d dB' %(20*np.log10(rmsplayback/rmssilence)))


