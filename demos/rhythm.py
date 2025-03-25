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
POWER_THRESH = 1e-6      # processing activated if signal power exceeds this threshold
CONCERT_PITCH = 440      # defining A4 (440 Hz)
WHITE_NOISE_THRESH = 0.2 # everything under WHITE_NOISE_THRESH*avg_energy_per_freq is cut off

WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ  # length of the window in seconds
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ           # length between two samples in seconds
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE      # frequency resolution of the FFT
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

ALL_NOTES = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"]
def find_closest_note(pitch):
    """
    Finds the closest note for a given pitch.
    Parameters:
      pitch (float): pitch in Hz.
    Returns:
      closest_note (str): e.g., A, G#, etc.
      closest_pitch (float): the frequency of the closest note.
    """
    i = int(np.round(np.log2(pitch / CONCERT_PITCH) * 12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = CONCERT_PITCH * 2**(i / 12)
    return closest_note, closest_pitch

HANN_WINDOW = np.hanning(WINDOW_SIZE)

def callback(indata, frames, time_info, status):
    """
    Callback function of the InputStream.
    Processes incoming audio, applies FFT and HPS,
    and detects when a note starts and ends.
    It records played notes as tuples: (note, start_time, duration)
    """
    # Define static variables on first call.
    if not hasattr(callback, "window_samples"):
        callback.window_samples = np.zeros(WINDOW_SIZE)
    if not hasattr(callback, "noteBuffer"):
        callback.noteBuffer = ["1", "2"]  # used for smoothing
    if not hasattr(callback, "recorded_notes"):
        callback.recorded_notes = []       # list to hold (note, start_time, duration) tuples
    if not hasattr(callback, "current_note"):
        callback.current_note = None       # current stable note
    if not hasattr(callback, "note_start_time"):
        callback.note_start_time = None    # time when the current note started

    if status:
        print(status)
        return

    if any(indata):
        # Update sliding window buffer.
        callback.window_samples = np.concatenate((callback.window_samples, indata[:, 0]))
        callback.window_samples = callback.window_samples[len(indata[:, 0]):]

        # Skip processing if overall signal power is too low.
        signal_power = (np.linalg.norm(callback.window_samples, ord=2)**2) / len(callback.window_samples)
        if signal_power < POWER_THRESH:
            # If no signal but a note was active, record its duration.
            if callback.current_note is not None:
                duration = time.time() - callback.note_start_time
                callback.recorded_notes.append((callback.current_note, callback.note_start_time, round(duration, 2)))
                callback.current_note = None
                callback.note_start_time = None
            os.system('cls' if os.name=='nt' else 'clear')
            print("Closest note: ...")
            print("Recorded notes:", callback.recorded_notes)
            return

        # Avoid spectral leakage by applying a Hann window.
        hann_samples = callback.window_samples * HANN_WINDOW
        magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples)//2])

        # Suppress mains hum: zero out frequencies below 62Hz.
        for i in range(int(62 / DELTA_FREQ)):
            magnitude_spec[i] = 0

        # Process each octave band: calculate average energy and suppress low-energy components.
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
        mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1/NUM_HPS),
                                  np.arange(0, len(magnitude_spec)), magnitude_spec)
        mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol)  # normalize

        # Calculate the Harmonic Product Spectrum (HPS).
        hps_spec = copy.deepcopy(mag_spec_ipol)
        for i in range(NUM_HPS):
            tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol)/(i+1)))],
                                         mag_spec_ipol[::(i+1)])
            if not any(tmp_hps_spec):
                break
            hps_spec = tmp_hps_spec

        max_ind = np.argmax(hps_spec)
        max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS

        closest_note, closest_pitch = find_closest_note(max_freq)
        max_freq = round(max_freq, 1)
        closest_pitch = round(closest_pitch, 1)

        # Update the note ring buffer for smoothing.
        callback.noteBuffer.insert(0, closest_note)
        callback.noteBuffer.pop()

        os.system('cls' if os.name=='nt' else 'clear')
        # Check if the ring buffer is stable.
        if callback.noteBuffer.count(callback.noteBuffer[0]) == len(callback.noteBuffer):
            stable_note = callback.noteBuffer[0]
            # If no note is currently active, start one.
            if callback.current_note is None:
                callback.current_note = stable_note
                callback.note_start_time = time.time()
            # If the stable note differs from the current one, record the previous note.
            elif callback.current_note != stable_note:
                duration = time.time() - callback.note_start_time
                callback.recorded_notes.append((callback.current_note, callback.note_start_time, round(duration, 2)))
                callback.current_note = stable_note
                callback.note_start_time = time.time()
            print(f"Closest note: {stable_note} {max_freq}/{closest_pitch}")
        else:
            print("Closest note: ...")

        # Always print the recorded notes.
        print("Recorded notes:", callback.recorded_notes)
    else:
        print("no input")

try:
    print("Starting HPS guitar tuner...")
    with sd.InputStream(channels=1, callback=callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
        while True:
            time.sleep(0.5)
except Exception as exc:
    print(str(exc))





