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
full_freq_spectrum = (scipy.fftpack.fft(audio_data)) # this returns an array of complex numbers
freq_spectrum = full_freq_spectrum[:len(full_freq_spectrum)//2] # just take the first half bc its mirrored (idk why)







print('done')

# COMPLEX NUMBERS AND FOURIER TRANSFORM CRASH COURSE (this explanation sux. sorry.)
   # if you want to understand this more, look at proofs for Taylor series, Euler's identity,
   # trig identities, the complex plane, and Fourier series. i dont really get it entirely tbh lol.

# Complex number: z = a + jb
# A complex number has a complex component (b) and a real component (a). j is the imaginary number sqrt(-1)
# complex number contains a lot of important info. it tells us the magnitude& phase of a certain freq component.
# Magnitude at that frequency is given by sqrt(a**2 + b**2)
# Phase, aka shift from sine, is given by arctain(b/a)
# The array is sorted by frequency. (not a math formula). the frequency is fs*k/N where k is the index
# Fourier transform is a way to identify which frequencies are contained in a periodic time domain signal
# fft is the fast fourier transform it uses polynomials or something to do the fourier transform faster idk
# the fft gives a symettric array of complex conjugates so we only need to look at one half of the array