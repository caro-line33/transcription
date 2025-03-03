import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

fs = 44100 # sampling frequency in Hertz
duration = 3 # samppling duration in seconds

''' Record the audio using sound device '''

def record_audio():
    print(f"starting {duration} second recording with a sampling frequency of {fs}")
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
        sd.wait() # wait for sound device to finish before continuing
        print("finished recording.")
        return audio.flatten() # convert to numpy array for easier mathematical processing
    except Exception as e:
        print(f"could not record. error: {e}")
        return None
    finally:
        sd.stop()

''' function for plotting recording in time domain '''
def plot_time_domain(audio):
    time = np.linspace(0, duration, num=len(audio))
    #start, stop, number of steps. (length of audio array = number of sample)
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio, label="Audio Signal", color='b')
    plt.xlabel("duration (seconds)")
    plt.ylabel("amplitude (normalized to be within -1 to 1 range)")
    plt.title("Time Domain Plot")
    plt.show()
    # this will output a plot that has an amplitude range of -1 to 1. this is the default
    # data type, dtype in sounddevice.rec() normalizes amplitude between these values. (not clipped)


''' function for plotting recording in freq domain '''
def plot_freq_domain(audio):
    audio_fft = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(audio), 1/fs)  # compute frequency bins (aka discrete frequencies) from audio_fft
    positive_freqs = freqs[:len(freqs) // 2]
    positive_magnitudes = np.abs(audio_fft[:len(audio_fft) // 2])

    plt.figure(figsize=(10, 4))
    plt.plot(positive_freqs, positive_magnitudes, color='b')
    plt.xlabel("frequency (hertz)")
    plt.ylabel("magnitude")
    plt.title("Frequency Domain Plot")
    plt.grid()
    plt.show()


''' function for choosing domaninant frequency '''
def get_dominant_frequency(audio, fs):
    n = len(audio)
    fft_data = np.fft.rfft(audio)
    fft_magnitude = np.abs(fft_data)
    frequencies = np.fft.rfftfreq(n, d=1/fs)
    dominant_index = np.argmax(fft_magnitude)
    dominant_frequency = frequencies[dominant_index]
    return dominant_frequency

''' code for note detection based on dominant frequency goes here'''
def closest_note(freq):
    print("hi")
    # write something here pls



''' user interface for recording & displaying values'''
while True:
    try:
        user_input = input("\n press 'r' to record, 'q' to quit: ").strip().lower()

        if user_input == "r":
            audio = record_audio()
            dominant_freq = round(get_dominant_frequency(audio, fs), 3)
            print(f"dominant freq: {dominant_freq} hz")
            plot_freq_domain(audio)

        elif user_input == "q":
            print("quitting program")
            break

        else:
            print("press 'r' to record or 'q' to quit.")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
        break

print("Program terminated")