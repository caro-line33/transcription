import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time

# General settings
SAMPLE_FREQ = 48000      # sample frequency in Hz
WINDOW_SIZE = 48000      # window size of the DFT in samples
WINDOW_STEP = 12000      # step size of window
NUM_HPS = 5              # max number of harmonic product spectrums
POWER_THRESH = 1e-6      # processing activated if signal power exceeds this
CONCERT_PITCH = 440      # reference pitch A4
WHITE_NOISE_THRESH = 0.2 # multiplier for noise suppression in each octave band
HPS_PEAK_THRESH = 1e-9   # minimum HPS peak amplitude to consider as a valid note

WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ  # window length in seconds
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ           # time between samples
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE      # frequency resolution of FFT
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
    i = int(np.round(np.log2(pitch / CONCERT_PITCH) * 12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = CONCERT_PITCH * 2**(i / 12)
    return closest_note, closest_pitch

HANN_WINDOW = np.hanning(WINDOW_SIZE)

def compute_hps(mag_spec_ipol):
    """
    Compute the Harmonic Product Spectrum (HPS) from the
    interpolated magnitude spectrum.
    """
    hps_spec = copy.deepcopy(mag_spec_ipol)
    for i in range(1, NUM_HPS):
        # Downsample by factor (i+1)
        downsampled = mag_spec_ipol[::(i+1)]
        length = min(len(hps_spec), len(downsampled))
        hps_spec = hps_spec[:length] * downsampled[:length]
    return hps_spec

def callback(indata, frames, time_info, status):
    """
    Callback function of the InputStream.
    Processes incoming audio, applies FFT, noise filtering,
    and then iteratively computes the HPS to detect multiple
    notes (polyphonic detection).
    """
    # Define static variables on first call.
    if not hasattr(callback, "window_samples"):
        callback.window_samples = np.zeros(WINDOW_SIZE)
    if not hasattr(callback, "noteBuffer"):
        callback.noteBuffer = []

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
            os.system('cls' if os.name=='nt' else 'clear')
            print("Closest note: ...")
            return

        # Apply a Hann window.
        hann_samples = callback.window_samples * HANN_WINDOW
        fft_result = scipy.fftpack.fft(hann_samples)
        magnitude_spec = np.abs(fft_result[:len(fft_result)//2])

        # Suppress mains hum: zero out frequencies below 62 Hz.
        for i in range(int(62 / DELTA_FREQ)):
            magnitude_spec[i] = 0

        # Process each octave band for noise suppression.
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
        norm_val = np.linalg.norm(mag_spec_ipol)
        if norm_val:
            mag_spec_ipol = mag_spec_ipol / norm_val
        else:
            os.system('cls' if os.name=='nt' else 'clear')
            print("Closest note: ...")
            return

        # Compute HPS on the original spectrum to determine the main peak.
        full_hps = compute_hps(mag_spec_ipol)
        orig_peak_amp = np.max(full_hps)
        if orig_peak_amp < HPS_PEAK_THRESH:
            os.system('cls' if os.name=='nt' else 'clear')
            print("Closest note: ...")
            return

        # Use a constant threshold for additional peaks (50% of main peak).
        amplitude_threshold = 0.75 * orig_peak_amp

        detected_notes = []
        # Create a modifiable copy for iterative peak detection.
        modified_mag_spec = mag_spec_ipol.copy()

        # Iteratively compute HPS and detect peaks.
        while True:
            hps_spec = compute_hps(modified_mag_spec)
            peak_index = np.argmax(hps_spec)
            peak_amp = hps_spec[peak_index]
            if peak_amp < amplitude_threshold:
                break
            # Compute frequency from the interpolated index.
            detected_freq = peak_index * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS
            note, note_pitch = find_closest_note(detected_freq)
            detected_notes.append((note, round(detected_freq,1), round(note_pitch,1)))
            # Zero out only the specific frequency bin where the peak was found.
            modified_mag_spec[peak_index] = 0

        os.system('cls' if os.name=='nt' else 'clear')
        if detected_notes:
            note_list = ", ".join([f"{n} ({f}Hz/{p}Hz)" for n, f, p in detected_notes])
            print("Detected notes: " + note_list)
        else:
            print("Closest note: ...")
    else:
        print("no input")

try:
    print("Starting polyphonic HPS tuner...")
    with sd.InputStream(channels=1, callback=callback,
                        blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
        while True:
            time.sleep(0.5)
except Exception as exc:
    print(str(exc))
