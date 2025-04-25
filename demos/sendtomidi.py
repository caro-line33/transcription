import sounddevice as sd
import numpy as np
import time
import os
from scipy.optimize import nnls
import mido

# ─── Set up MIDI ────────────────────────────────────────────────────────────────
PORT_NAME = 'loopMIDI Port 1'
midi_out = mido.open_output(PORT_NAME)

# ─── Load your dictionaries ─────────────────────────────────────────────────────
dictionary = np.load("notes_matrix.npy")
dict_norms = np.linalg.norm(dictionary, axis=1, keepdims=True)   # (N_notes,1)
dictionary_unit = dictionary / (dict_norms + 1e-12)

# ─── Audio & FFT settings ──────────────────────────────────────────────────────
sample_freq = 8400
fft_length  = 8400
threshold   = 1e-4
blocksize   = 1024

# ─── Utility functions ─────────────────────────────────────────────────────────
def process(input_signal):
    windowed    = input_signal * np.hanning(len(input_signal))
    fft_result  = np.fft.rfft(windowed)
    spectrum    = np.abs(fft_result)[26:]
    return spectrum

def top_n_idx(v, n=10):
    idx = np.argpartition(v, -n)[-n:]
    return idx[np.argsort(v[idx])[::-1]]

def tonality_score(buf):
    spectrum = process(buf)
    mean     = np.average(spectrum)
    max_sq   = np.max(spectrum)**2
    return max_sq/(mean+1e-12), spectrum

def idx_to_midi(i):
    return 24 + i

note_names = [
    # c1...c6
    *["{}{}".format(n, o) for o in range(1, 7) for n in ["c","cs","d","ds","e","f","fs","g","gs","a","as","b"]],
]

# ─── The callback ──────────────────────────────────────────────────────────────
def callback(indata, frames, time_info, status):
    if status:
        print(status)
        return

    # initialize on first call
    if not hasattr(callback, "_buf"):
        callback._buf        = np.zeros(fft_length, dtype=float)
        callback._write_idx  = 0
        callback._ema        = 0.0
        callback._prev_ema   = 0.0
        callback._dec_count  = 0
        callback._note_buff  = {}
        callback._current    = None

    # ring buffer fill
    buf = callback._buf
    idx = callback._write_idx
    audio = indata[:,0]
    end = idx + len(audio)
    if end < fft_length:
        buf[idx:end] = audio
    else:
        first = fft_length - idx
        buf[idx:]      = audio[:first]
        buf[:end-fft_length] = audio[first:]
    callback._write_idx = end % fft_length

    # silence check
    energy = np.sum(buf**2)
    os.system('cls' if os.name=='nt' else 'clear')
    if energy < threshold:
        if callback._current is not None:
            midi_out.send(mido.Message('note_off',
                                       note=idx_to_midi(callback._current),
                                       velocity=64))
            print(f"NOTE OFF (silence): {note_names[callback._current]}")
            callback._current = None
            callback._note_buff.clear()
        # reset EMA so we detect a fresh start next time
        callback._prev_ema = callback._ema
        return

    # tonality & EMA
    score, spectrum = tonality_score(buf)
    callback._ema = 0.88 * callback._ema + 0.12 * score
    print(f"tonality score: {score:.2f}, EMA: {callback._ema:.2f}")

    # decay counter
    delta = callback._ema - callback._prev_ema
    callback._dec_count = callback._dec_count+1 if delta < -19 else 0

    # note detection
    if callback._ema >= 250 and callback._dec_count < 3:
        cos_sims = dictionary_unit @ (spectrum / (np.linalg.norm(spectrum)+1e-12))
        candidates = top_n_idx(cos_sims, n=5)
        top_note   = candidates[0]
        print("Top match:", top_note, note_names[top_note])

        # vote buffer
        callback._note_buff[top_note] = callback._note_buff.get(top_note, 0) + 1
        # purge others
        for k in list(callback._note_buff):
            if k != top_note:
                del callback._note_buff[k]

        # only fire when stable ≥2 frames
        if callback._note_buff[top_note] >= 2:
            if callback._current != top_note:
                # turn off old
                if callback._current is not None:
                    midi_out.send(mido.Message('note_off',
                                               note=idx_to_midi(callback._current),
                                               velocity=64))
                    print(f"NOTE OFF: {note_names[callback._current]}")
                # turn on new
                midi_out.send(mido.Message('note_on',
                                           note=idx_to_midi(top_note),
                                           velocity=64))
                print(f"NOTE ON:  {note_names[top_note]} (MIDI {idx_to_midi(top_note)})")
                callback._current = top_note

    callback._prev_ema = callback._ema

# ─── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        with sd.InputStream(channels=1,
                            callback=callback,
                            samplerate=sample_freq,
                            blocksize=blocksize):
            print("Listening… press Ctrl-C to stop")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        midi_out.close()
