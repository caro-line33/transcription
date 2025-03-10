import sounddevice as sd
import numpy as np
import time
import os
import scipy.fftpack

sample_freq = 48000
threshold = 0.02
duration = 0.1

def callback(indata, frames, time, status):
    if status:
        print(status)
        return
    

    energy = np.sum(indata ** 2)
    

    os.system('cls' if os.name == 'nt' else 'clear')
    

    if energy > threshold:
        print("I can hear you")
    else:
        print("I cannot hear you")

try:
    with sd.InputStream(channels=1, callback=callback, samplerate=sample_freq, blocksize=int(sample_freq * duration)):
        while True:
            time.sleep(0.1)
except Exception as e:
    print(e)


