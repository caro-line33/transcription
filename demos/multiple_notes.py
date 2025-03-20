import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time
from collections import deque

sample_freq = 48000 # number of samples taken per second
window_size = 48000


def callback(indata, frames, time_info, status):
    if status:
        print(status)
        return
    full_frequency = scipy.fftpack.fft(indata.flatten())
    frequency = full_frequency[:len(full_frequency)//2]
    freq_res = sample_freq/window_size
    max_downsampling = 5
    new_res = freq_res/max_downsampling
    magnitude = np.abs(frequency)
    original_indices = np.arange(len(magnitude))  
    more_indices = np.arange(0, len(magnitude), new_res)
    magnitude_interp = np.interp(more_indices, original_indices, magnitude)
    hps_spec = magnitude_interp.copy()

    for i in range(2, (max_downsampling+1)):  
      downsampled = magnitude_interp[::i]
      min_length = min(len(hps_spec), len(downsampled))
      hps_spec = hps_spec[:min_length] * downsampled[:min_length]
    
    max_ind = np.argmax(hps_spec)
    fundamental_frequency = max_ind * new_res 
    os.system('cls' if os.name == 'nt' else 'clear')
    print(fundamental_frequency)
    
try:
    with sd.InputStream(channels=1, callback=callback, samplerate=sample_freq, blocksize=window_size):
        while True:
            time.sleep(0.01)
except Exception as e:
    print(e)