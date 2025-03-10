import sounddevice as sd
import numpy as np
import time
import os
import scipy.fftpack

# Constants
sample_freq = 48000  # Samples per second
threshold = 0.02     # Power threshold to define sound/silence
duration = 0.05      # Duration of each audio chunk

def assign_note(frequency):
    a0 = 27.5  # Lowest note on piano
    notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    number_of_steps = int(np.round((np.log2(frequency/a0)) * 12)) 
    octave = (9 + number_of_steps) // 12 
    note = notes[number_of_steps % 12]
    return note, octave

def hps(data):
    # Apply FFT
    freq_spectrum = scipy.fftpack.fft(data.flatten())
    magnitude_spectrum = np.abs(freq_spectrum)[:len(freq_spectrum)//2]  # Half spectrum

    # Harmonic Product Spectrum (HPS)
    harmonics = 5
    hps_spectrum = magnitude_spectrum.copy()
    for i in range(1, harmonics):
        scaled_spectrum = np.interp(np.arange(len(hps_spectrum)), np.arange(0, len(hps_spectrum), i + 1), magnitude_spectrum[::(i + 1)])
        hps_spectrum *= scaled_spectrum  # Element-wise multiplication

    peak_freq = np.argmax(hps_spectrum)  # Find index of peak frequency
    peak_frequency = peak_freq * (sample_freq / len(data))  # Convert index to frequency
    return peak_frequency

# The callback function is called by sounddevice when audio is available
def callback(indata, frames, time, status):
    if status:
        print(status)
        return

    energy = np.sum(indata ** 2)
    os.system('cls' if os.name == 'nt' else 'clear')

    if energy > threshold:
        # Calculate the frequency and assign the note
        frequency = hps(indata)
        note, octave = assign_note(frequency)
        print(f"Detected Note: {note}{octave}, Frequency: {frequency:.2f} Hz")
    else:
        print("No sound detected")

# This block runs forever, processing audio until interrupted
try:
    with sd.InputStream(channels=1, callback=callback, samplerate=sample_freq, blocksize=int(sample_freq * duration)):
        while True:
            time.sleep(0.1)  # Keeps the program running
except Exception as e:
    print(e)

