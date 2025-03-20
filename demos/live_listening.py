import sounddevice as sd
import numpy as np
import time
import os

sample_freq = 48000 # number of samples taken per second
threshold = 0.02 # power threshold to define what is a sound and what is silence
duration = 0.1 # duration of audio chunk in seconds

def callback(indata, frames, time_info, status):
    # this is the function that is called by sd.InputStream each time a new audio chunk is available
    # status is an instance of sounddevice.CallbackFlags. if there are any that means an error has occured
    # if statement below prevents the code from continuing if errors are encountered
    if status:
        print(status)
        return
    
    # calculate total energy of the audio chunk by calculating the sum of squares
    # **2 on an array squares every value, np.sum adds up those squares
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


