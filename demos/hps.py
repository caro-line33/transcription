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


def spectral_crest(input_signal):
    fft_result = np.fft.rfft(input_signal)
    spectrum = abs(fft_result)
    mean = np.average(spectrum)
    max_sq = np.max(spectrum)**2
    score = max_sq/mean
    return score, spectrum


def fast_hps(input_spectrum):
    

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
    
    # compute energy over the whole buffer
    energy = np.sum(buf**2)
    os.system('cls' if os.name=='nt' else 'clear')

    spectral_score, spectrum = spectral_crest(buf)
    # time domain silence blocking 
    if energy < threshold:
        print("no sound detected...")
        score = 0
    else:
        score = spectral_score
    callback._ema = 0.8 * callback._ema + 0.2 * score
    print(f"tonality score: {score:.2f}, EMA: {callback._ema:.2f}")

    if callback._ema >= 200:
        

        


if __name__ == "__main__":
    try:
        with sd.InputStream(channels=1, callback=callback, samplerate=sample_freq, blocksize=blocksize):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped")
    except Exception as e:
        print(str(e))