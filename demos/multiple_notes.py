''' in the file hps.py we record an audio clip and then use harmonic product spectrum to identify the 
fundamental frequency in the recording. here we use hps to detect several notes at once. this can be used when
chords and stuff are played.'''

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
harmonics = 5 # Scale the spectrum down 5 times for 5 harmonics
for i in range(1, harmonics):
    scaled_spectrum = np.interp(np.arange(len(hps_spectrum)), np.arange(0, len(hps_spectrum), i + 1), magnitude_spectrum[::(i + 1)])
    hps_spectrum *= scaled_spectrum  # Multiply the current HPS spectrum with the scaled spectrum

# Find the peak frequency in the HPS spectrum
peak_freq = np.argmax(hps_spectrum)  # Find index of peak magnitude
peak_frequency = peak_freq * (fs/len(audio_data))  # Convert index to frequency

print(f"fundamental frequency: {peak_frequency} Hz")
print('done')