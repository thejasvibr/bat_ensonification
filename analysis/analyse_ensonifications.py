# -*- coding: utf-8 -*-
"""
Set of functions to calculate target strength,
and impulse response of the ensonified bat

Created on Sun Jan 22 16:08:38 2017

The input data requirements

@author: tbeleyur
"""
import numpy as np
import scipy.signal as signal
import scipy.fftpack as spyfft

folder_location = 'C://Users//tbeleyur//Documents//bat_ensonification_data//30_degrees//'

def calc_target_strength(rec_empty_room, rec_w_bat, rec_distance):
    '''
    calculates target strength of bat at a particular distance
    and provides a dB target strength estimate at 1 metre.

    It does this by subtracting the dB rms of the reference
    empty room recording with the dB rms of the recording w
    the bat in the room.

    Inputs:
    rec_empty_room : np.array. recording w empty room
    rec_w_bat : np.array. recording w bat
    rec_distance: float. bat-mic distance in metres

    Output:
    ts_dB : float. target strength of the bat in dB

    '''
    rms_empty_room = np.std(rec_empty_room)
    rms_w_bat = np.std(rec_w_bat)

    # CHECK IF YOU SHOULD USE 20 DB OR 10 DB !!
    ts_dB = 20*np.log10(rms_w_bat/rms_empty_room)

    if not(rec_distance ==1.0) :
        return(ts_dB)
    else :
        # taken from http://www.sengpielaudio.com/calculator-distance.htm
        ts_dB_1m = ts_dB + 20*np.log10(1.0/rec_distance)

        return(ts_dB_1m)

def calc_IR(rec_empty_room,rec_w_bat,window_size=1024):
    '''
    CALCULATES IMPULSE RESPONSE OF THE BAT AT A GIVEN POSITION AND DISTANCE :

    Inputs:
    rec_empty_room : np.array. empty room recording.
    rec_w_bat: np.array. recording with bat
    window_size = integer. window size required for output impulse response.

    Outputs:
    impulse_response: np.array. output impulse response of required window length

    '''

    half_window = window_size/2
    ir_correlation = np.correlate (rec_empty_room,rec_w_bat,'same')
    max_cor = np.argmax(abs(ir_correlation))

    impulse_response = ir_correlation[max_cor-half_window:max_cor+half_window]



    return(impulse_response)


if __name__ == '__main__':
    print('hi')

    rec_empty_room = np.random.normal(0,0.1,100000)
    echo_filter = np.hstack((np.zeros(100),[0.2],np.zeros(100),[0.1])    )

    b,a = signal.butter(8,[0.5],btype='highpass')
    w_bat_echo = signal.lfilter(echo_filter,1,rec_empty_room) * 0.05
    rec_w_bat =rec_empty_room*0.01 +  signal.lfilter(b,a,w_bat_echo)


    ir = calc_IR(rec_w_bat,rec_empty_room,1024)

    amp_ir = np.std(rec_w_bat)/np.std(ir) * ir

    # testing if the ir actuallyr ecreates the recorded signal :
    filt_sig = signal.lfilter(amp_ir,1,rec_empty_room )

    # checking out the target strength of the mock bat :
    ts_mock_bat = calc_target_strength(rec_empty_room,rec_w_bat,1)




    plt.figure(1)
    empty_room_fresp = 20*np.log10(abs(spyfft.fft(rec_empty_room)))
    empty_room_fresp = empty_room_fresp[:rec_empty_room.size/2]
    plt.plot(empty_room_fresp)
    plt.plot()

    w_bat_fresp = 20*np.log10(abs(spyfft.fft(rec_w_bat)))
    w_bat_fresp = w_bat_fresp[:w_bat_fresp.size/2]
    plt.plot(w_bat_fresp)

    # recreating the signal FFT :

    sig_ir_fresp = 20*np.log10(abs(spyfft.fft( filt_sig )))
    sig_ir_fresp = sig_ir_fresp[:filt_sig.size/2]
    plt.plot(sig_ir_fresp)

    plt.figure(2)
    plt.plot(empty_room_fresp-np.max(empty_room_fresp) )
    plt.plot(w_bat_fresp-np.max(w_bat_fresp) )
    plt.plot(sig_ir_fresp - np.max(sig_ir_fresp) )

    # plotting the waveforms :
    plt.figure(3)
    plt.plot(rec_w_bat)
    plt.plot(filt_sig )


    # check how similar the spectra of the filtered and w bat signal are :
    plt.figure(4)
    plt.plot(sig_ir_fresp-w_bat_fresp)
    plt.title('dB difference in frequency content: convolved signal - w bat signal')








