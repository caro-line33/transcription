import sounddevice as sd
import numpy as np
import scipy.fftpack  # scipy fft is faster than numpy fft

fs = 44800  # sampling frequency
duration = 2  # seconds
harmonics = 5  # Number of harmonics to use for HPS

# Recording audio
print('Starting recording...')
audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)  # sd.rec returns a numpy array
sd.wait()  # wait until the recording is done to start the next step
print('Recording finished.')

# Perform FFT to get frequency spectrum
freq_spectrum = scipy.fftpack.fft(audio_data.flatten())  # Flatten to convert to 1D array
magnitude_spectrum = np.abs(freq_spectrum)[:len(freq_spectrum)//2]  # Use just the first half.

# Harmonic Product Spectrum (HPS) Calculation
hps_spectrum = magnitude_spectrum.copy()
for i in range(1, harmonics):
    # Downsample and multiply the spectrums together
    scaled_spectrum = np.interp(np.arange(len(hps_spectrum)), np.arange(0, len(hps_spectrum), i + 1), magnitude_spectrum[::(i + 1)])
    hps_spectrum *= scaled_spectrum

# Function to detect peaks
def detect_peaks(spectrum, threshold_ratio=0.5):
    detected_peaks = []
    while True:
        peak_freq = np.argmax(spectrum)  # Find index of peak magnitude
        peak_value = spectrum[peak_freq]

        if peak_value == 0:
            break

        peak_frequency = peak_freq * (fs / len(audio_data))  # Convert index to frequency
        detected_peaks.append((peak_frequency, peak_value))

        # Set the peak to zero to avoid detecting the same peak again
        spectrum[peak_freq] = 0

        # Find the second highest peak
        second_peak_spectrum = np.copy(spectrum)
        second_peak_spectrum[peak_freq] = 0
        second_peak_freq = np.argmax(second_peak_spectrum)
        second_peak_value = second_peak_spectrum[second_peak_freq]
        
        # Check if the second peak is significant (e.g., > half of the first peak)
        if second_peak_value > peak_value * threshold_ratio:
            continue
        else:
            break

    return detected_peaks

# Detect peaks in the HPS spectrum
detected_peaks = detect_peaks(hps_spectrum, threshold_ratio=0.5)

# Print the detected frequencies and their magnitudes
print("Detected frequencies and magnitudes:")
for freq, value in detected_peaks:
    print(f"Frequency: {freq} Hz, Magnitude: {value}")
