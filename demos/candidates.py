import numpy as np

# load the array from disk
dictionary = np.load("notes_matrix.npy")

import sounddevice as sd
import numpy as np
import time
import os
import librosa
import statistics

sample_freq = 8400
fft_length = 8400
threshold = 1e-4
blocksize = 1024

import numpy as np

def top_n_idx(v, n=5):
    v = np.asarray(v)  
    # find the n largest indices
    idx = np.argpartition(v, -n)[-n:]
    # sort them descending by value
    idx = idx[np.argsort(v[idx])[::-1]]
    # add 26 to each index
    return idx


def process(input_signal):
    windowed = input_signal * np.hanning(len(input_signal))
    fft_result = np.fft.rfft(windowed)
    spectrum_full = abs(fft_result)
    spectrum = spectrum_full[26:]
    return spectrum

def callback(indata, frames, time_info, status):
    if status:
        print(status)
        return
    
    # one-time init of the circular buffer and state tracking
    if not hasattr(callback, "_buffer"):
        callback._buffer = np.zeros(fft_length, dtype=float)
        callback._write_idx = 0
        callback._score_idx = 0
        callback._score_buffer = np.zeros(fft_length//3, dtype=float)
        callback._ema = 0.0
        callback._prev_ema   = 0.0
        callback._dec_count  = 0.0
    
    # Get ring buffer and add current block to it
    buf = callback._buffer
    idx = callback._write_idx
    audio = indata[:, 0]
    n = len(audio)
    end = idx + n
    if end < fft_length:
        buf[idx:end] = audio
    else:
        first = fft_length - idx
        buf[idx:] = audio[:first]
        buf[:end-fft_length] = audio[first:]
    callback._write_idx = end % fft_length

    os.system('cls' if os.name=='nt' else 'clear')
    signal = process(buf)
    scores = [np.dot(dictionary[i], signal) for i in range(60)]
    candidates = top_n_idx(scores)
    print(candidates)
    
if __name__ == "__main__":
    try:
        with sd.InputStream(channels=1, callback=callback, samplerate=sample_freq, blocksize=blocksize):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped")
    except Exception as e:
        print(str(e))