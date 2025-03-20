import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time
from collections import deque

sample_freq = 48000 # number of samples taken per second
threshold = 0.02 # power threshold to define what is a sound and what is silence
duration = 0.1 # duration of audio chunk in seconds

def callback(indata, frames, time_info, status):
    if status:
        print(status)
        return
    full_frequency = scipy.fftpack.fft(indata.flatten())
    frequency = full_frequency[:len(full_frequency)//2]
    magnitude = np.abs(frequency)
    max_mag = np.argmax(magnitude)
    bin_range = 1/duration
    dominant_frequency = max_mag*bin_range+(bin_range/2)
    os.system('cls' if os.name == 'nt' else 'clear')
    print(dominant_frequency)
    
try:
    with sd.InputStream(channels=1, callback=callback, samplerate=sample_freq, blocksize=int(sample_freq * duration)):
        while True:
            time.sleep(0.1)
except Exception as e:
    print(e)
