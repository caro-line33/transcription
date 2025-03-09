''' This file demonstrates the Harmonic Product Spectrum, a way to detect the fundamental frequency.
    The Harmonic Product Spectrum (HPS) is a way to find the pitch of a sound by looking for its 
    fundamental frequency — the main "note" you hear. 
    It works by:

    1. Calculating the frequency spectrum of the sound (using a Fourier Transform).
    2. Making smaller, "downsampled" versions of the spectrum — each representing the 2nd, 3rd, etc. harmonics.
    3. Multiplying these spectra together — so the harmonics overlap and boost the fundamental frequency's peak.
    4. The highest peak after multiplying shows the pitch since true harmonics will align at the fundamental 
       frequency, cutting through noise and false peaks.

    After finding the fundamental frequency of the sample using Harmonic Product Spectrum, we can assign it
       to a note, demonstrated by note_id.py '''

import sounddevice as sd
import numpy as np
import scipy.fftpack # scipy fft is faster than numpy fft

fs = 44800 # sampling frequency
duration = 2

# sd.rec(frames, samplerate, channels, dtype, callback, blocking, device, channels, blocking)
# only frames, samplerate, channels are required as arguments.

print('starting recording!')
audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1) # sd.rec returns a numpy array
sd.wait # wait until the recording is done to start the next step



print('done')
