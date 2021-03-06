# -*- coding: utf-8 -*-
"""
Script which calculates the cIR for the speaker
to get a flat frequency response

Created on Tue Jan 26 Jan 12:22 2017

User records sound from a flat-response microphone
like a GRAS 1/4th inch and as output the compensatory
IR of the particular speaker is generated


Watch out for the speaker amplification settings while playing back noise signal
Start with lowest amplification level and gently increase the playback volume
based on the signal to noise ratio of the recordings got.


IR and cIR calculations based on the measIR MATLAB script written by Holger R Goerlitz

@author: tbeleyur
"""
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.fftpack as spyfft
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.signal as signal
from scipy.interpolate import InterpolatedUnivariateSpline
plt.rcParams['agg.path.chunksize'] = 10000
import scipy.io.wavfile as wav

np.random.seed(612) # to recreate the same playback signals at any point of time.

### generate noise signal :

def gen_gaussian_noise(num_samples,mean,sdev):
    # generates gaussian distribution of sample values

    gaussian_noise = np.random.normal(mean,sdev,num_samples)
       
    return(gaussian_noise)

def filter_signal(input_signal,order,freq_fraction,filter_type):
    # lowpass,highpass or bandpass es a signal with a butterworth of
    # given order
    # also returns b,a
    b,a = signal.butter(order,freq_fraction,btype=filter_type)
    filtered_signal = signal.lfilter(b,a,input_signal)
    return(filtered_signal,[b,a])

def add_ramps(half_ramp_samples,orig_signal):
    # adds up and down ramps to the original signal
    full_window = np.hamming(2*half_ramp_samples)
    up_ramp = full_window[:half_ramp_samples]
    down_ramp = full_window[-half_ramp_samples:]

    sig_w_ramp = np.copy(orig_signal)
    sig_w_ramp[:half_ramp_samples] *= up_ramp
    sig_w_ramp[-half_ramp_samples:] *= down_ramp

    return(sig_w_ramp)

def zero_pad(num_samples,sig):
    
    # adds num_samples number of zeros on either size of the signal 
    sig_zeropad = np.pad(sig,(num_samples,num_samples),'constant',constant_values=(0,0))
    
    return(sig_zeropad)


def get_impulse_response(input_signal,rec_signal,ir_length,FS,exp_delaysamples):

    input_sig_flat = np.ndarray.flatten(input_signal)
    rec_sig_flat = np.ndarray.flatten(rec_signal[exp_delaysamples:])

    print('signals being cross-correlated')
    cross_cor = np.correlate( input_sig_flat, rec_sig_flat, 'same')

    # get the index of maximum correlation to align the impulse response properly
    max_corr =  np.argmax(abs(cross_cor))

    # choose the cross corr along a particular window size
    impulse_resp = cross_cor[max_corr-int(ir_length/2) :max_corr+ int(ir_length/2)]

    return( [impulse_resp])

def freq_deconv(rec_sig,target_sig):
    '''
    deconvolves an original signal with the ideal dirac pulse response 
    '''
        
    norm_rec_sig = ( np.std(target_sig)/(np.std(rec_sig)) ) * rec_sig
    
    target_freq = spyfft.fft(target_sig)
    sig_freq = spyfft.fft(norm_rec_sig)
    
    deconv_freq = target_freq/sig_freq
    
    return(deconv_freq)
    

def get_pwr_spec(sig):
    
    powspec = 20*np.log10(abs(spyfft.fft(sig)))[:sig.size/2]
    
    return(powspec)
    
    
    
    
def calc_cIR(impulse_resp,ir_length,ba_list):
    # create a Dirac pulse (which has aLL frequencies)
    dirac_pulse = np.zeros(ir_length)
    dirac_pulse[ir_length/2 -1] = 1

    b = ba_list[0]
    a = ba_list[1]
    # filter the frequencies for the dirac pulse and the recorded IR:
    dirac_pulse_filtered = signal.lfilter(b,a,dirac_pulse)

    # standardised the rms of the impulse response w the filtered dirac_pulse
    impulse_resp_norm = impulse_resp*(np.std(dirac_pulse_filtered) / np.std(impulse_resp) )

    print('FFTs being calculated...')
    # calculate the fft's of both filtered signals :
    filt_dpulse_fft = spyfft.fft(dirac_pulse_filtered)
    filt_iresp_fft = spyfft.fft(impulse_resp_norm)

    # now divide the all frequency signal w the some-frequency signal :
    # to get the cIR :
    cIR_fft = filt_dpulse_fft / filt_iresp_fft

    # calculate the iFFT to get a compensatory IR filter :
    cIR = spyfft.ifft(cIR_fft).real # here HRG shifts array circularly ..why ?
    

    cIR_final = np.roll(cIR,int(ir_length/2) )
    
    cIR_filt = signal.lfilter(b,a,cIR_final)

    return(cIR_filt)

flatten = np.ndarray.flatten # one line function assignment

def oned_fft_interp(new_freqs,fft_freqs,fft_var,interp_type='linear',kvalue=3):



    if interp_type =='cubic':
        interp_spline = InterpolatedUnivariateSpline(flatten(fft_freqs),flatten( 20*np.log10(abs(fft_var))),k=kvalue)
        interp_power_spectrum = flatten(interp_spline(new_freqs))
        return(interp_power_spectrum)
    else:
        interp_power_spectrum= interp1d(flatten(fft_freqs), flatten( 20*np.log10(abs(fft_var)) ),kind = interp_type )
        return(interp_power_spectrum(new_freqs))



if __name__ == '__main__':
    
    durn_pbk = 2
    FS = 192000
    numramp_samples = 0.1*FS
    mic_speaker_dist = 0.74 # in meters
    vsound = 320 # in meters/sec
    delay_time = float(mic_speaker_dist/vsound)
    ir_length = 512
    input_channels = [2,9]
    output_channels = [2,1]
    total_num_samples = int(durn_pbk*FS)
    pass_frequency = np.array([10000.0 ,90000.0])
    nyquist_freq = FS/2
    
    device_num = 38
    #### generate playbacks :
    
    gaussian_noise = gen_gaussian_noise(total_num_samples,0,0.2)
    filt_gaussian_noise,ba_list = filter_signal(gaussian_noise, 8, pass_frequency/nyquist_freq,'bandpass')
    
    # trigger spike to get the playback delay :
    trigger_sig = np.zeros(gaussian_noise.size)
    trigger_sig [0] = 1.0
    
    def execute_cIR_calculation():
    
        pbk_sig =  add_ramps( numramp_samples ,filt_gaussian_noise )
        
        final_pbk = np.column_stack((trigger_sig,pbk_sig ))
        
        print('raw sound being played now...')
        rec_sound = sd.playrec(final_pbk,FS,input_mapping=input_channels,output_mapping=output_channels ,dtype='float',device = device_num)
        sd.wait()
        
        print('signal processing happening now...')
        
        delay_samples = int(delay_time *FS)
        
        intfc_pbk_delay = np.argmax(rec_sound[:,0]) # audio interface playback delay
        
        irparams = get_impulse_response(pbk_sig,rec_sound[intfc_pbk_delay:,1],ir_length,FS,delay_samples)
        
        print('impulse and frequency response being calculated now...')
        
        cir = calc_cIR(irparams[0],ir_length,ba_list)
        
        print('signal being convolved with cIR now ')
        
        corrected_sig = signal.lfilter(cir,1,pbk_sig)
        
        corrected_hp_sig = signal.lfilter(ba_list[0],ba_list[1],corrected_sig)
        
        print('corrected_sound being played now...')
        amp_dB = -( 20*np.log10(np.std(corrected_hp_sig)) - 20*np.log10(np.std(pbk_sig)) )   # in dB
        
        amp_factor = 10**(amp_dB/20.0)
        
        # amplified corrected signal
        amped_corr_sig = amp_factor*corrected_hp_sig
        
        corrected_pbk = np.column_stack((trigger_sig, amped_corr_sig ))
        
        rec_corrected_sound = sd.playrec(corrected_pbk,FS,input_mapping=input_channels,output_mapping=output_channels ,dtype='float',device = device_num)
        sd.wait()
        
        print('plotting taking place...')
        
        plt.figure(1)
        plt.plot(cir)
        
        # smoothing over a Kilohertz range
        min_plot_freq = np.min(pass_frequency)
        max_plot_freq = int(FS/2)
        freq_range = max_plot_freq - min_plot_freq
        smoothing_freqs = np.linspace(min_plot_freq,max_plot_freq,freq_range/500)
        
        
        plt.figure(3)
        
        plt.subplot(411)
        plt.plot(rec_sound[intfc_pbk_delay:,1])
        plt.title('original recorded signal')
        
        plt.subplot(412)
        orig_fft = spyfft.rfft(rec_sound[delay_samples:,1])
        num_freqs= np.linspace(0,FS/2,orig_fft.size)
        plt.plot(num_freqs,20*np.log10(abs(orig_fft)))
        plt.title('FFT original recorded sound')
        plt.ylim(-80,30)
        
        sm_fft_orig= oned_fft_interp(smoothing_freqs,num_freqs,orig_fft,'cubic')
        plt.plot(smoothing_freqs,sm_fft_orig)
        
        
        plt.subplot(413)
        plt.plot(rec_corrected_sound[delay_samples:,1])
        plt.title('cIR X original sound recorded sound')
        
        
        plt.subplot(414)
        crct_sig_fft = spyfft.rfft(rec_corrected_sound[delay_samples:,1])
        num_freqs_crct= np.linspace(0,FS/2,crct_sig_fft.size)
        plt.plot(num_freqs_crct,20*np.log10(abs(crct_sig_fft)))
        plt.title('FFT: with cIR recorded sound')
        plt.ylim(-80,30)
        
        sm_fft= oned_fft_interp(smoothing_freqs,num_freqs_crct,crct_sig_fft,'cubic')
        plt.plot(smoothing_freqs,sm_fft)
        
        plt.figure(4)
        sm_fft= oned_fft_interp(smoothing_freqs,num_freqs_crct,crct_sig_fft)
        plt.plot(smoothing_freqs,sm_fft-np.max(sm_fft),'r')
        sm_fft_orig= oned_fft_interp(smoothing_freqs,num_freqs,orig_fft,'cubic')
        plt.plot(smoothing_freqs,sm_fft_orig-np.max(sm_fft_orig),'g')
        
        print('corrected_recording : smoothed frequency spectrum dB range is %d' %(np.max(sm_fft)-np.min(sm_fft)  ))
        print('original_recording : frequency spectrum dB range is %d' %(np.max(sm_fft_orig)-np.min(sm_fft_orig)  ))
        return(amped_corr_sig)
        
        #fft_res = spyfft.rfft(ccor)
        #plt.plot(np.linspace(0,FS/2,2048),20*np.log10(abs(fft_res)))
        
        
    
    
#    print('\n running cIR now ')
#    corrected_signal = execute_cIR_calculation()
#    prev_cor = np.copy(corrected_signal)
#    c_file = 'C:\\Users\\tbeleyur\\Documents\\noise_playback.WAV'
#    try:    
#        wav.write(c_file,192000,corrected_signal)
#        print('wav file saved succesfully')
#    except:
#        print('error in saving file ')
#        
        
    
