# -*- coding: utf-8 -*-
"""
output speaker IR corrected lienar sweeps from the viva speakers


Created on Feb 15 2017

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
import calc_cIR as ir_funcs
import sys,os
import datetime as dt
sys.stdout.flush()
sys.path.append(os.path.realpath('..'))
import playback_saving_funcs as pbksave



# location to where the generated data is saved to as numpy arrays
target_folder = 'C:\\Users\\tbeleyur\\Desktop\\ensonification_data\\2017_03_13\\playback_sweeps\\2ms\\'


# playback and recording details :
durn = 0.002
FS = 192000
num_samples = int(durn*FS)
ramp_durn = 0.0005
ramp_samples = int(ramp_durn * FS)
silence_durn = 0.5
silence_samples = int(FS*silence_durn)
dist_mic_speaker = 1.0 # distance in metres
vsound = 320.0
trans_delay_samples = int((dist_mic_speaker/vsound)*FS)

start_freq = 96000.0
end_freq = 20000.0
freq_sweep = np.linspace(start_freq,end_freq,num_samples)
time = np.linspace(0,durn,num_samples)


sweep = signal.chirp(time,start_freq,np.max(time),end_freq,'log')


hp_freq = 10000

device_list = sd.query_devices()
tgt_dev_name = 'ASIO Fireface USB'
tgt_dev_bool = [tgt_dev_name in each_device['name'] for each_device in device_list]
tgt_ind = int(np.argmax(np.array(tgt_dev_bool)))

dev_in_ch = [2,10]
dev_out_ch = [2,1]

hp_b, hp_a = signal.butter(8,[float(hp_freq)/FS],'highpass')




orig_sig = sweep #linear_sweep #ir_funcs.gen_gaussian_noise(num_samples,0,0.2)

orig_sig = signal.lfilter(hp_b,hp_a,orig_sig)
ramp_orig = ir_funcs.add_ramps(ramp_samples,orig_sig)
orig_zeropad = ir_funcs.zero_pad(silence_samples,ramp_orig)

sync_channel = np.zeros(ramp_orig.size)
sync_channel[0] = 1
sync_zeropad = ir_funcs.zero_pad(silence_samples,sync_channel)

orig_playback = np.column_stack( (sync_zeropad,orig_zeropad) )

rec_sound = sd.playrec(orig_playback,samplerate= FS, input_mapping=dev_in_ch, output_mapping= dev_out_ch, device = tgt_ind)
sd.wait()



rec_sync_index = np.argmax(abs(rec_sound[:,0]))
post_sync_rec = rec_sound[rec_sync_index:,1]

pbk_sync = np.argmax(abs(orig_playback[:,0]))
post_sync_orig = orig_playback[pbk_sync:,1]
post_sync_orig = post_sync_orig[:post_sync_rec.size]

print('signal correlation happening now')
align_cor = np.correlate(post_sync_rec,post_sync_orig,'same')

print('')

mid_point_cor = align_cor.size/2 -1
align_index =  np.argmax(abs(align_cor)) - mid_point_cor

aligned_rec = post_sync_rec[align_index:align_index+num_samples]
aligned_orig = post_sync_orig[:num_samples]


tgt_sig = np.zeros(aligned_rec.size)
tgt_sig [tgt_sig.size/2 -1 ] =1


comp_freq = ir_funcs.freq_deconv(aligned_rec,ramp_orig)

orig_freq = spyfft.fft(aligned_orig)
conv_freq = comp_freq*orig_freq
conv_sig = spyfft.ifft(conv_freq).real

plt.figure(2)
plt.title('Spectra: Original signal, recorded signal, compensatory signal  ')
plt.ylabel('Power, dB')

digital_sig, = plt.plot(np.linspace(0,96,aligned_orig.size/2),ir_funcs.get_pwr_spec(aligned_orig),label='original signal')
uncomp_rec, = plt.plot(np.linspace(0,96,aligned_rec.size/2),ir_funcs.get_pwr_spec(aligned_rec),label='uncompensated recordings')

comp_freqs, = plt.plot(np.linspace(0,96,comp_freq.size/2),20*np.log10(abs(comp_freq[:comp_freq.size/2])),label ='compensated spectrum')

plt.legend(handles = [digital_sig,uncomp_rec,comp_freqs])

ramp_convsig = ir_funcs.add_ramps(ramp_samples,conv_sig)
sync_ch_convsig = np.zeros(ramp_convsig.size)
sync_ch_convsig[0] = 1

zerop_convsig = ir_funcs.zero_pad(silence_samples,ramp_convsig)
zerop_sync_ch_convsig = ir_funcs.zero_pad(silence_samples,sync_ch_convsig)

hp_convsig = signal.lfilter(hp_b,hp_a,zerop_convsig)
rms_norm_convsig = np.std(ramp_orig)/np.std(ramp_convsig) * zerop_convsig


conv_playback = np.column_stack((zerop_sync_ch_convsig,rms_norm_convsig))


print('compensated playback occuring now')
# rms_norm_convsig
conv_rec = sd.playrec(conv_playback ,samplerate=FS,input_mapping=dev_in_ch,output_mapping=dev_out_ch,device=tgt_ind)
sd.wait()

# plot the convolved playback recording taking into account the time
# required for sound to reach after playback

pbk_delay =np.argmax(abs(conv_playback[:,0]))
total_delay = pbk_delay + trans_delay_samples



plt.figure(3)
plt.title('Power spectrum of digital signal and speaker IR compensated signal')
plt.xlabel('Frequency, KHz')
plt.ylabel('Power, dB (rel. max dB value)')

fft_convrec = ir_funcs.get_pwr_spec( conv_rec[align_index:align_index+num_samples,1] )
freqs_convrec = np.linspace(0,96,fft_convrec.size)

fft_orig = ir_funcs.get_pwr_spec(ramp_orig)
freqs_orig = np.linspace(0,96,fft_orig.size)

new_freqs = np.linspace(20,96,76)

intp_val_convrec = ir_funcs.oned_fft_interp(new_freqs,freqs_convrec,fft_convrec,'linear')
intp_val_orig = ir_funcs.oned_fft_interp(new_freqs,freqs_orig,fft_orig,'linear')

convrec_plot, = plt.plot(new_freqs,intp_val_convrec-np.max(intp_val_convrec),'y*-',label ='speaker IR compensated')
digital_sig, = plt.plot(new_freqs,intp_val_orig-np.max(intp_val_orig),'b*-',label='digital signal')
plt.legend(handles = [convrec_plot,digital_sig],bbox_to_anchor=(0.5,0.2),loc=1,borderaxespad=0.)

plt.figure(1)
plt.figure(1)
plt.plot(rec_sound,label='uncompensated recording')
plt.plot(conv_rec[:,1],label='convolved recording')
plt.title('recording of raw & convolved recording')
plt.legend()


plt.figure(4)
window_length = 2048
ir_max_point = np.argmax(abs(align_cor))
plt.title('Impulse Response centered on max corr point : %d sample window length'%window_length)
plt.plot(align_cor[ir_max_point-window_length/2:ir_max_point+window_length/2])

plt.figure(5)
# smooth the power spectrum a bit to understand what's happening:
window_size = 10
moving_average_window = np.ones(window_size)/window_size
smoothed_spectrum = np.convolve(fft_convrec,moving_average_window,'same')
plt.plot(np.linspace(0,96,smoothed_spectrum.size),smoothed_spectrum)
plt.grid()
plt.title('Smoothed power spectrum post convolution %s window size'%window_size)
plt.xlabel('KHz')
plt.ylabel('dB Power')

# column stack the digital playback signal and the compensated signal for future reference :
time_stamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M')

np.save(target_folder+'wcIRplayback_recording_' + time_stamp,conv_rec)
np.save(target_folder+'cIR_conv_signal_'+time_stamp,rms_norm_convsig)
np.save(target_folder+'orig_signal_'+time_stamp,orig_sig)

saved_sound = pbksave.save_rec_file(conv_rec[:,1],FS,target_folder+'GRAS_''wcIRplayback_recording_' + time_stamp+'.WAV')







