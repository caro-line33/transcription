import sounddevice as sd
import numpy as np
import time
import os
import numpy as np
from scipy.optimize import nnls

dictionary = np.load("notes_matrix.npy")
dict_norms = np.linalg.norm(dictionary, axis=1, keepdims=True)   # shape (N_notes,1)
dictionary_unit = dictionary / (dict_norms + 1e-12) 


sample_freq = 8400
fft_length = 8400
threshold = 1e-4
blocksize = 1024

def process(input_signal):
    windowed = input_signal * np.hanning(len(input_signal))
    fft_result = np.fft.rfft(windowed)
    spectrum_full = abs(fft_result)
    spectrum = spectrum_full[26:]
    return spectrum

def top_n_idx(v, n=10):
    v = np.asarray(v)  
    idx = np.argpartition(v, -n)[-n:]
    idx = idx[np.argsort(v[idx])[::-1]]
    return idx

def tonality_score(input_signal):
    fft_result = np.fft.rfft(input_signal)
    spectrum_full = abs(fft_result)
    spectrum = spectrum_full[26:]
    mean = np.average(spectrum)
    max_sq = np.max(spectrum)**2
    score = max_sq/mean
    return score, spectrum



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
    
    # compute energy over the whole buffer
    energy = np.sum(buf**2)
    os.system('cls' if os.name=='nt' else 'clear')

    # time domain silence blocking 
    if energy < threshold:
        print("no sound detected...")
        score = 0
    else:
        score, spectrum = tonality_score(buf)
    callback._ema = 0.88 * callback._ema + 0.12 * score
    print(f"tonality score: {score:.2f}, EMA: {callback._ema:.2f}")

    delta = callback._ema - callback._prev_ema 
    if delta < -19:
        callback._dec_count += 1
    else:
        callback._dec_count = 0

    # ONLY IF IT PASSES ALL THE TESTS CAN WE DO PROCESSING...
    if (callback._ema >= 250) and (callback._dec_count < 3):
        print(f"i think there's music playing...")
        signal = process(buf)
        scores = [np.dot(dictionary[i], signal) for i in range(60)]

        # compute all cosine similarities in one matrix‐vector multiply
        sig_norm = signal / (np.linalg.norm(signal) + 1e-12)
        cos_sims = dictionary_unit @ sig_norm

        # pick top 5
        candidates = top_n_idx(cos_sims, n=5)
        print("Top matches (cosine):", candidates)

        # NNLS solve using top 5- this part is generating an error due to dimension mismatch!
        b = signal
        A = []
        for index in candidates:
            A.append(dictionary[index])
        x, residual = nnls(A, b)
        print(x)
            
    

    print(f"delta ema: {delta:.2f}")
    print(f"dec count: {callback._dec_count:.2f}")
    
    callback._prev_ema = callback._ema
        

if __name__ == "__main__":
    try:
        with sd.InputStream(channels=1, callback=callback, samplerate=sample_freq, blocksize=blocksize):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped")
    except Exception as e:
        print(str(e))