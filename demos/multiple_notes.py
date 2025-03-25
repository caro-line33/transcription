import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time

# General settings that can be changed by the user
SAMPLE_FREQ = 48000      # sample frequency in Hz
WINDOW_SIZE = 48000      # window size of the DFT in samples
WINDOW_STEP = 12000      # step size of window
NUM_HPS = 5              # max number of harmonic product spectrums
POWER_THRESH = 1e-6      # tuning is activated if the signal power exceeds this threshold
CONCERT_PITCH = 440      # defining A4 (440 Hz)
WHITE_NOISE_THRESH = 0.2 # multiplier for noise suppression in each octave band
HPS_PEAK_THRESH = 1e-5   # if the HPS peak amplitude is below this threshold, consider it noise

WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ  # length of the window in seconds
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ           # time between samples
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE      # frequency resolution of the FFT
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
def find_closest_note(pitch):
    """
    Finds the closest musical note for a given pitch.
    Parameters:
      pitch (float): pitch in Hz.
    Returns:
      closest_note (str): e.g., A, G#, etc.
      closest_pitch (float): the frequency of the closest note.
    """
    i = int(np.round(np.log2(pitch/CONCERT_PITCH)*12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = CONCERT_PITCH * 2**(i/12)
    return closest_note, closest_pitch

HANN_WINDOW = np.hanning(WINDOW_SIZE)

def callback(indata, frames, time_info, status):
    """
    Callback function of the InputStream.
    Processes the incoming audio, applies FFT and HPS,
    and uses both power and HPS peak thresholds to avoid
    detecting a note when there is only noise.
    """
    # Define static variables on the first call.
    if not hasattr(callback, "window_samples"):
        callback.window_samples = np.zeros(WINDOW_SIZE)
    if not hasattr(callback, "noteBuffer"):
        callback.noteBuffer = ["1", "2"]

    if status:
        print(status)
        return

    if any(indata):
        # Update the sliding window buffer.
        callback.window_samples = np.concatenate((callback.window_samples, indata[:, 0]))
        callback.window_samples = callback.window_samples[len(indata[:, 0]):]

        # Skip processing if overall signal power is too low.
        signal_power = (np.linalg.norm(callback.window_samples, ord=2)**2) / len(callback.window_samples)
        if signal_power < POWER_THRESH:
            os.system('cls' if os.name=='nt' else 'clear')
            print("Closest note: ...")
            return

        # Apply a Hann window to reduce spectral leakage.
        hann_samples = callback.window_samples * HANN_WINDOW
        fft_result = scipy.fftpack.fft(hann_samples)
        magnitude_spec = np.abs(fft_result[:len(fft_result)//2])

        # Suppress mains hum: zero out frequencies below 62 Hz.
        for i in range(int(62/DELTA_FREQ)):
            magnitude_spec[i] = 0

        # Process each octave band to suppress noise.
        for j in range(len(OCTAVE_BANDS)-1):
            ind_start = int(OCTAVE_BANDS[j] / DELTA_FREQ)
            ind_end = int(OCTAVE_BANDS[j+1] / DELTA_FREQ)
            if ind_end > len(magnitude_spec):
                ind_end = len(magnitude_spec)
            band = magnitude_spec[ind_start:ind_end]
            if len(band) == 0:
                continue
            avg_energy_per_freq = np.sqrt((np.linalg.norm(band, ord=2)**2) / len(band))
            for i in range(ind_start, ind_end):
                if magnitude_spec[i] <= WHITE_NOISE_THRESH * avg_energy_per_freq:
                    magnitude_spec[i] = 0

        # Interpolate the spectrum for higher resolution in HPS.
        x_orig = np.arange(len(magnitude_spec))
        x_interp = np.arange(0, len(magnitude_spec), 1/NUM_HPS)
        mag_spec_ipol = np.interp(x_interp, x_orig, magnitude_spec)
        mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol)  # Normalize

        # Calculate the Harmonic Product Spectrum (HPS).
        hps_spec = copy.deepcopy(mag_spec_ipol)
        for i in range(NUM_HPS):
            tmp_hps_spec = np.multiply(
                hps_spec[:int(np.ceil(len(mag_spec_ipol)/(i+1)))],
                mag_spec_ipol[::(i+1)]
            )
            if not any(tmp_hps_spec):
                break
            hps_spec = tmp_hps_spec

        # Check if the HPS peak amplitude is significant.
        max_ind = np.argmax(hps_spec)
        max_amp = hps_spec[max_ind]
        if max_amp < HPS_PEAK_THRESH:
            os.system('cls' if os.name=='nt' else 'clear')
            print("Closest note: ...")
            return

        max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS
        closest_note, closest_pitch = find_closest_note(max_freq)
        max_freq = round(max_freq, 1)
        closest_pitch = round(closest_pitch, 1)

        # Update the note ring buffer for smoothing.
        callback.noteBuffer.insert(0, closest_note)
        callback.noteBuffer.pop()

        os.system('cls' if os.name=='nt' else 'clear')
        if callback.noteBuffer.count(callback.noteBuffer[0]) == len(callback.noteBuffer):
            print(f"Closest note: {closest_note} {max_freq}/{closest_pitch}")
        else:
            print("Closest note: ...")
    else:
        print('no input')

try:
    print("Starting HPS guitar tuner...")
    with sd.InputStream(channels=1, callback=callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
        while True:
            time.sleep(0.5)
except Exception as exc:
    print(str(exc))

