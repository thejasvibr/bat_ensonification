# -*- coding: utf-8 -*-
"""

Created on Tue Nov 22 14:06:28 2016

Script that analyses each chunk of noise playback and plots the frequency spectrum
of each chunk as a separate line

@author: tbeleyur
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
from scipy.io import wavfile

def findsyncpulse(syncsignal):
    '''
    finds the absolute maximum of the synchronising pulse

    input:
    syncsignal: np.array (N,).

    output:
    syncindex: np.integer. - index of the signal with maximum abs value

    '''
    abssignal=abs(syncsignal)
    syncindex=np.where(abssignal==max(abssignal))[0][0]

    return(syncindex)

def  normalisesignal(signal,bitrecording):

    maxval=2**bitrecording -1
    normfactor=maxval/2.0

    normsignal=signal/normfactor

    return(normsignal)


def signalpostpulse(recsound,syncindex):

    '''
    cuts out the recorded sounds from the sync pulse onwards and returns the signal without the previous portion
    '''
    cutsound=recsound[syncindex:]

    return(cutsound)



def calculateIR(originalsignal,recsignal):


    pass



def findrecordingdelay(recordedsig,internalsig,correlnlength,speaker2micdist,vsound,FS):
    '''
    finds the time lag in number of samples between the start of speaker playback and mic recording
    Note: the function works reasonably fast if the correlnlengths are kept <100000 samples

    Inputs:
        recordedsig: np.array. the array with mike recordings
        internalsig: np.array. the array with the internal signal which was played through the signal
        speaker2micdist: float. distance between mic and speaker in meters
        vsound: float. speed of sound
        FS: integer. sampling rate


    '''
    traveltime=speaker2micdist/vsound

    delaysamples=int(np.around(traveltime*FS))

    recfirstplayback=recordedsig[:correlnlength+delaysamples]
    internalfirstplayback=internalsig[:recfirstplayback.shape[0]]
    sigcor=np.correlate(recfirstplayback,internalfirstplayback,mode='same')

    lagindex=  np.argmax(sigcor) - sigcor.shape[0]/2.0
    if lagindex <=0:
        print('error in the cross-correlation. Mic recording begins before the internal signal')
    else:

        return(int(lagindex) )

def highpassfilter(signal,hpfreq,FS):

    a,b=scipy.signal.butter(8,hpfreq/FS,'high')
    hp_signal=scipy.signal.lfilter(a,b,signal)

    return(hp_signal)


def extractplaybacks(recsig,playbacksamples,silencesample,delayindex,numplaybacks):
    '''
    cuts out the noise playbacks and assigns them as separate np.arrays
    Input:
    internalsig:np.array. he full internal reference signal that was recorded - from the sync pulse onwards
    recsig: np.array. the recorded signal
    playbacksamples: int.number of samples the playback lasts for
    silencesamples: int. number of samples the silence between the playbacks lasts for.
    numplaybacks: int. number of playbacks in the recording

    Output:
    recplaybacks: np.array w numplaybacks x playbackduration  dimensions. each row has one recorded playback

    '''

    recplaybacks=np.zeros((playbacksamples,numplaybacks))

    timeshiftedsig=recsig[delayindex:]

    istart=0
    for eachplayback in range(numplaybacks):

        recplaybacks[:,eachplayback]=timeshiftedsig[istart:istart+playbacksamples]

        istart+=silencesamples+playbacksamples

    return(recplaybacks)



multiplewavreader=lambda file_name: wavfile.read(file_name)



if __name__== '__main__':

    # tests for the various functions :
    plt.rcParams['agg.path.chunksize'] = 10000

     # now testing it with a real file ! :
    FS=192000
    targetdir='C:\\Users\\tbeleyur\\Documents\\speaker_directionality_measurements\\23_Nov_recordings_w)speakercIR_GRASmic\\round_1\\'
    internal_name='internal_record.wav'
    micrec_name='GRAS_MICROPHONE.wav'
    syncrec_name='sync_signal.wav'
    filenames=[internal_name,micrec_name,syncrec_name]

    playbacksamples=FS*3.0
    silencesamples=FS*1.5
    numplaybacks=19


    fullpaths=map(lambda x: targetdir+x,filenames)
    recsound=map(lambda filename: wavfile.read(filename)[1],fullpaths)




    #testing findsyncpulse:
    #expected answer to be 10
    fakesyncsignal=np.hstack( (np.zeros(10),np.array([-1,1]),np.zeros(100) ))
    #findsyncpulse(fakesyncsignal)

    # testing calcmaxcorr:
    #calcmaxcorr(recsig,internalsig,FS,mic2spkrdistance,vsound)
    noise=np.random.normal(0,0.1,1000)
    recsig=np.hstack((np.zeros(300),noise ))
    internalsig=np.hstack((np.zeros(100),noise,np.zeros(200)     ))
    vsound=330
    dist=0.35
    FS=192000

    #calcmaxcorr(recsig,internalsig,FS,dist,vsound)



    # testing findrecordingdelay:
    noissig=np.random.normal(0,0.1,10000)
    backgnoise=np.random.normal(0,0.01,noissig.shape[0])
    startsilence=np.zeros(1000)
    spkr2micdist=0.375
    vsound=340
    samplerate=192000
    delaysamples=int(np.around((spkr2micdist/vsound)*FS))


    testplayback=np.hstack((noissig,np.zeros(delaysamples) ))
    recordedplayback=np.hstack((np.zeros(delaysamples),noissig ))
    noisyrecordedplayback=np.hstack((np.zeros(delaysamples),noissig )) + np.random.normal(0,0.01,recordedplayback.shape[0])
    sigcor=np.correlate(recordedplayback,testplayback,mode='same')

    #findrecordingdelay(recordedsig,internalsig,startsilence,playbacksamples,speaker2micdist,vsound,FS)

    delayindex=findrecordingdelay(recordedplayback,testplayback,1000,spkr2micdist,vsound,samplerate)

    # checking the correlation with a noisy signal too !!
    noisydelayindex=findrecordingdelay(noisyrecordedplayback,testplayback,2000,spkr2micdist,vsound,samplerate)


    # testing it with real recordings :
    hp=1000
    a,b=scipy.signal.butter(8,hp/192000.0,'high')
    hp_sound=scipy.signal.lfilter(a,b,recsound[1])

    syncindex=findsyncpulse(recsound[2])
    internalsig=signalpostpulse(recsound[0],syncindex)

    hp_recsig=signalpostpulse(hp_sound,syncindex)
    postpulse_lagindex=findrecordingdelay(hp_recsig,internalsig,100000,0.375,330,192000)
#
############# extractplaybacks(recsig,playbacksamples,silencesample,delayindex,numplaybacks)

    allpbks=extractplaybacks(hp_recsig,576000,288000,220,19)

