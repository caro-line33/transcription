import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

def record_audio(duration=3, fs=44100):
    print("recording started")
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
        sd.wait()
        print("recording finished")
        sd.stop()
        return audio.flatten(), fs
    except Exception as e:
        print(f"error: {e}")

def plot_waveform(audio, fs, duration):
    time = np.linspace(0, duration, num=len(audio))
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio, label="Audio Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Recorded Audio Waveform")
    plt.legend()
    plt.grid()
    plt.show()

while True:
    user_input = input("\nPress r to record:").strip().lower()
    if user_input == "r":
        audio, fs = record_audio()
        plot_waveform(audio, fs, duration=3)