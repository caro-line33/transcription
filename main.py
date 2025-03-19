import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time
from collections import deque

# Global parameters
SAMPLE_FREQ = 48000          # sample frequency in Hz
WINDOW_SIZE = 48000          # window size of the DFT in samples
WINDOW_STEP = 12000          # step size between windows (new samples per callback)
NUM_HPS = 5                  # max number of harmonic product spectrums
POWER_THRESH = 1e-6          # threshold for signal power
CONCERT_PITCH = 440          # reference pitch (A4)
WHITE_NOISE_THRESH = 0.2     # fraction for noise thresholding

DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE       # frequency resolution
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

def find_closest_note(pitch):
    """Map a frequency (pitch in Hz) to the closest musical note."""
    i = int(np.round(np.log2(pitch / CONCERT_PITCH) * 12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = CONCERT_PITCH * 2 ** (i / 12)
    return closest_note, closest_pitch

# Precompute the Hann window once
HANN_WINDOW = np.hanning(WINDOW_SIZE)

def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        print("\033[H\033[J", end="")

def callback(indata, frames, time_info, status):
    """
    Process new audio samples using a circular buffer for efficient sliding window updates.
    """
    # Initialize circular buffer and pointer on first call.
    if not hasattr(callback, "circular_buffer"):
        callback.circular_buffer = np.zeros(WINDOW_SIZE)
        callback.buffer_index = 0
    # Initialize the note ring buffer for smoothing
    if not hasattr(callback, "noteBuffer"):
        callback.noteBuffer = deque(maxlen=3)

    if status:
        print(status)
        return

    if np.any(indata): # only proceed if at least one value is nonzero
        new_samples = indata[:, 0]
        num_new = len(new_samples)
        
        # Insert new_samples into the circular buffer at the current index
        if callback.buffer_index + num_new <= WINDOW_SIZE:
            # No wrap-around needed
            callback.circular_buffer[callback.buffer_index:callback.buffer_index + num_new] = new_samples
            callback.buffer_index += num_new
        else:
            # Wrap-around: fill the remainder of the buffer and then the beginning.
            remaining = WINDOW_SIZE - callback.buffer_index
            callback.circular_buffer[callback.buffer_index:] = new_samples[:remaining]
            callback.circular_buffer[:num_new - remaining] = new_samples[remaining:]
            callback.buffer_index = num_new - remaining
        
        # Reconstruct the contiguous window for FFT:
        # The window is the circular buffer starting from buffer_index to end, then from beginning to buffer_index.
        if callback.buffer_index == 0:
            window_samples = callback.circular_buffer.copy()
        else:
            window_samples = np.concatenate(
                (callback.circular_buffer[callback.buffer_index:], callback.circular_buffer[:callback.buffer_index])
            )

        # Compute signal power; skip processing if below threshold
        signal_power = np.linalg.norm(window_samples)**2 / WINDOW_SIZE
        if signal_power < POWER_THRESH:
            clear_screen()
            print("Closest note: ...")
            return

        # Apply Hann window to reduce spectral leakage
        hann_samples = window_samples * HANN_WINDOW
        fft_result = scipy.fftpack.fft(hann_samples)
        half = len(fft_result) // 2
        magnitude_spec = np.abs(fft_result[:half])

        # Suppress frequencies below 62 Hz (vectorized)
        idx_low = int(62 / DELTA_FREQ)
        magnitude_spec[:idx_low] = 0

        # Process each octave band to suppress noise
        for j in range(len(OCTAVE_BANDS) - 1):
            ind_start = int(OCTAVE_BANDS[j] / DELTA_FREQ)
            ind_end = int(OCTAVE_BANDS[j+1] / DELTA_FREQ)
            if ind_end > len(magnitude_spec):
                ind_end = len(magnitude_spec)
            band = magnitude_spec[ind_start:ind_end]
            if band.size == 0:
                continue
            avg_energy = np.sqrt(np.sum(band**2) / band.size)
            threshold = WHITE_NOISE_THRESH * avg_energy
            band = np.where(band <= threshold, 0, band)
            magnitude_spec[ind_start:ind_end] = band

        # Interpolate the magnitude spectrum for better resolution in HPS
        x_orig = np.arange(len(magnitude_spec))
        x_interp = np.arange(0, len(magnitude_spec), 1/NUM_HPS)
        mag_spec_ipol = np.interp(x_interp, x_orig, magnitude_spec)
        norm = np.linalg.norm(mag_spec_ipol)
        if norm:
            mag_spec_ipol /= norm

        # Compute the Harmonic Product Spectrum (HPS)
        hps_spec = mag_spec_ipol.copy()
        for i in range(1, NUM_HPS):
            downsampled = mag_spec_ipol[::(i+1)]
            length = min(len(hps_spec), len(downsampled))
            tmp_hps_spec = hps_spec[:length] * downsampled[:length]
            if not np.any(tmp_hps_spec):
                break
            hps_spec = tmp_hps_spec

        max_ind = np.argmax(hps_spec)
        max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS
        closest_note, closest_pitch = find_closest_note(max_freq)
        max_freq = round(max_freq, 1)
        closest_pitch = round(closest_pitch, 1)

        # Update note ring buffer for smoothing
        callback.noteBuffer.appendleft(closest_note)
        clear_screen()
        if len(callback.noteBuffer) < callback.noteBuffer.maxlen or \
           callback.noteBuffer.count(callback.noteBuffer[0]) != len(callback.noteBuffer):
            print("Closest note: ...")
        else:
            print(f"Closest note: {closest_note} {max_freq}/{closest_pitch}")
    else:
        print("no input")

try:
    print("Starting HPS guitar tuner...")
    with sd.InputStream(channels=1, callback=callback,
                        blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
        while True:
            time.sleep(0.5)
except Exception as exc:
    print(str(exc))

