''' This file demonstrates the Harmonic Product Spectrum, a way to detect the fundamental frequency.
    The Harmonic Product Spectrum (HPS) is a way to find the pitch of a sound by looking for its 
    fundamental frequency — the main "note" you hear. 
    It works by:

    1. Calculating the frequency spectrum of the sound (using a Fourier Transform).
    2. Making smaller, downscaled versions of the spectrum — one for each harmonic (there's infinite harmonics)
    we can just calculate for 5 of them to make this computationally lighter.
    3. Multiplying these spectra together — so the harmonics overlap and boost the fundamental frequency's peak.
    4. The highest peak after multiplying shows the pitch since true harmonics will align at the fundamental 
       frequency, cutting through noise and false peaks.

    After finding the fundamental frequency of the sample using Harmonic Product Spectrum, we can assign it
       to a note, demonstrated by note_id.py '''

import sounddevice as sd
import numpy as np
import scipy.fftpack  # scipy fft is faster than numpy fft

fs = 44800  # sampling frequency
duration = 2  # seconds

# sd.rec(frames, samplerate, channels, dtype, callback, blocking, device, channels, blocking)
# only frames, samplerate, channels are required as arguments.

print('starting recording!')
audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)  # sd.rec returns a numpy array
sd.wait()  # wait until the recording is done to start the next step

# Perform FFT to get frequency spectrum
freq_spectrum = scipy.fftpack.fft(audio_data.flatten())  # Flatten to convert to 1D array
magnitude_spectrum = np.abs(freq_spectrum)[:len(freq_spectrum)//2]  # Use just the first half.

# HPS Calculation
hps_spectrum = magnitude_spectrum.copy()
harmonics = 5
for i in range(1, harmonics):
    resampled_spectrum = np.interp(np.arange(len(hps_spectrum)), np.arange(0, len(hps_spectrum), i + 1), magnitude_spectrum[::(i + 1)])
    hps_spectrum *= resampled_spectrum  # Multiply the current HPS spectrum with the resampled spectrum

# Find the peak frequency in the HPS spectrum
peak_freq = np.argmax(hps_spectrum)  # Find index of peak magnitude
peak_frequency = peak_freq * (fs/len(audio_data))  # Convert index to frequency

print(f"fundamental frequency: {peak_frequency} Hz")
print('done')

# math review
   # complex numbers tell us the magnitude and phase shift
   # we can find the corresponding frequency by the complex number's index in the array
   # hps works by scaling down the sample repeatedly since harmonics are integer multiples of fundamental freq