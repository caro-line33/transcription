import sounddevice as sd
import numpy as np
import time
import os

sample_freq = 8400
fft_length  = 8400
blocksize   = 1024

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

    # —— one-time init ——
    if not hasattr(callback, "_init"):
        callback._buffer        = np.zeros(fft_length, dtype=float)
        callback._write_idx     = 0
        callback._ema           = 0.0
        callback._note_on       = False
        callback._collected     = []        # list of spectra
        callback._init          = True

    # —— ring-buffer write ——
    buf = callback._buffer
    idx = callback._write_idx
    audio = indata[:,0]
    n = len(audio)
    end = idx + n
    if end < fft_length:
        buf[idx:end] = audio
    else:
        first = fft_length - idx
        buf[idx:] = audio[:first]
        buf[:end-fft_length] = audio[first:]
    callback._write_idx = end % fft_length

    # —— compute EMA score as before ——
    crest, spectrum = tonality_score(buf)  # use last 2048 samples for speed
    score = crest
    callback._ema = 0.88*callback._ema + 0.12*score

    # —— on/off logic (simple threshold + hysteresis) ——
    ON_THR  = 60
    OFF_THR = 20
    if not callback._note_on:
        if callback._ema > ON_THR:
            callback._note_on = True
            callback._collected.clear()
            print("▶ Detected NOTE ON- starting collection")
    else:
        if callback._ema < OFF_THR:
            callback._note_on = False
            print("■ Detected NOTE OFF- stopping collection")
            # —— process collected spectra —— 
            if callback._collected:
                # stack and take median (or mean)
                all_specs = np.vstack(callback._collected)
                note_vector = np.mean(all_specs, axis=0)
                # save to disk
                np.save("c6.npy", note_vector)
                os.system('c1s' if os.name=='nt' else 'clear')
                print("Saved note_vector.npy (length {})".format(len(note_vector)))
                raise sd.CallbackStop()
        else:
            # still in-note: collect this spectrum
            callback._collected.append(spectrum)

    os.system('cls' if os.name=='nt' else 'clear')
    print(f"EMA={callback._ema:.1f}  {'ON' if callback._note_on else 'off'}  collected={len(callback._collected)}")

if __name__=="__main__":
    with sd.InputStream(channels=1, callback=callback,
                        samplerate=sample_freq, blocksize=blocksize):
        print("Play a note when ready…")
        while True:
            time.sleep(0.1)
