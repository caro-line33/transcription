""" 
This takes in audio chunks one by one and sees if the energy contained in the sample is above a 
certain threshold. If not, then it says "I cannot hear you". if so, it says "I can hear you".

This file demonstrates how audio can be continuously processed in real time. In the previous code, we had 
to record an entire clip before processing it. Here we use InputStream from sounddevice in order to process 
it continuously.
 """

import sounddevice as sd # getting audio from computer microphone
import numpy as np # use numpy for mathematical processing, faster than built in functions
import time # create pause in program
import os # compatibility with different operating systems (for clearing the terminal later)

sample_freq = 48000 # number of samples taken per second
threshold = 0.02 # power threshold to define what is a sound and what is silence
duration = 0.1 # duration of audio chunk in seconds

def callback(indata, frames, time, status):
    
    # status is an instance of sounddevice.CallbackFlags
    # if there are any that means an error has occured
    # this if statement prevents the code from continuing if errors are encountered
    if status:
        print(status)
        return
    
    
    # calculate total energy of the audio chunk by calculating the sum of squares
    # **2 on an array squares every value, np.sum adds up those squares
    energy = np.sum(indata ** 2)
    
    # clear the terminal based on operating system, so that the continuous outputs don't flood the terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # if the energy calculated is greater than the defined threshold of silence then ...
    if energy > threshold:
        print("I can hear you")
    else:
        print("I cannot hear you")


try:
    #try opening an audio stream using sd.InputStream
    with sd.InputStream(channels=1, callback=callback, samplerate=sample_freq, blocksize=int(sample_freq * duration)):
        # the callback parameter is the callback function above. it is called every time a new 
        # chunk of audio is available. the size of a chunk is defined by blocksize.
        while True:
            time.sleep(0.1)  # Keeps the program running, pausing to reduce CPU usage. 
except Exception as e:
    print(e)


