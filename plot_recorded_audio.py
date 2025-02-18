import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

fs = 44100 # sampling frequency in Hertz
duration = 3 # samppling duration in seconds

def record_audio():
    print(f"starting {duration} second recording with a sampling frequency of {fs}")
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
        sd.wait() # wait for sound device to finish before continuing
        print("alright im done.")
        return audio.flatten() # convert to numpy array for easier mathematical processing
    except Exception as e:
        print(f"could not record. error: {e}")
        return None
    finally:
        sd.stop()

def plot_time_domain(audio):
    if audio is None:
        print("No audio data to plot.")
        return
    time = np.linspace(0, duration, num=len(audio))
    #start, stop, number of steps. (length of audio array = number of sample)
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio, label="Audio Signal", color='b')
    plt.xlabel("duration (seconds)")
    plt.ylabel("amplitude (normalized to be within -1 to 1 range)")
    plt.title("Time Domain Plot of Recorded Audio")
    plt.show()
    # this will output a plot that has an amplitude range of -1 to 1. this is the default
    # data type, dtype in sounddevice.rec() normalizes amplitude between these values. (not clipped)



def plot_freq_domain(audio):
    print("freq domain plot")
    return None

while True:
    try:
        user_input = input("\nPress 'r' to record, 'q' to quit: ").strip().lower()

        if user_input == "r":
            audio = record_audio()
            plot_time_domain(audio)
            plot_freq_domain()

        elif user_input == "q":
            print("Exiting program.")
            break

        else:
            print("Invalid input. Press 'r' to record or 'q' to quit.")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
        break
