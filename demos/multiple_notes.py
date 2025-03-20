import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time
from scipy.signal import find_peaks

sample_freq = 48000 
window_size = 48000

def find_closest_note(pitch):
    """Map a frequency (pitch in Hz) to the closest musical note."""
    CONCERT_PITCH = 440
    ALL_NOTES = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"]
    i = int(np.round(np.log2(pitch/CONCERT_PITCH)*12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = CONCERT_PITCH * 2**(i/12)
    return closest_note, closest_pitch

def callback(indata, frames, time_info, status):
    if status:
        print(status)
        return

    full_frequency = scipy.fftpack.fft(indata.flatten())
    frequency = full_frequency[:len(full_frequency)//2]
    
    freq_res = sample_freq / window_size
    
    max_downsampling = 5
    new_res = freq_res / max_downsampling
    
    magnitude = np.abs(frequency)
    original_indices = np.arange(len(magnitude))
    more_indices = np.arange(0, len(magnitude), new_res)
    magnitude_interp = np.interp(more_indices, original_indices, magnitude)
    
    hps_spec = magnitude_interp.copy()
    for i in range(1, max_downsampling):
        downsampled = magnitude_interp[::(i+1)]
        min_length = min(len(hps_spec), len(downsampled))
        hps_spec = hps_spec[:min_length] * downsampled[:min_length]
    
    peaks, properties = find_peaks(hps_spec, height=np.max(hps_spec)*0.3)
    
    detected_notes = []
    for peak in peaks:
        freq = peak * new_res
        note, ref_pitch = find_closest_note(freq)
        detected_notes.append((note, freq, ref_pitch))
    
    os.system('cls' if os.name == 'nt' else 'clear')
    if detected_notes:
        print("Detected notes:")
        for note, freq, ref_pitch in detected_notes:
            print(f"{note}: {freq:.1f} Hz (ref: {ref_pitch:.1f} Hz)")
    else:
        print("No notes detected")
    
try:
    print("Starting polyphonic note detection...")
    with sd.InputStream(channels=1, callback=callback, samplerate=sample_freq, blocksize=window_size):
        while True:
            time.sleep(0.01)
except Exception as e:
    print(e)
