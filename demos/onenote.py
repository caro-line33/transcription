import sounddevice as sd
import numpy as np
import time
import os
import numpy as np
from scipy.optimize import nnls
import mido

dictionary = np.load("notes_matrix.npy")
dict_norms = np.linalg.norm(dictionary, axis=1, keepdims=True)   # shape (N_notes,1)
dictionary_unit = dictionary / (dict_norms + 1e-12) 
midi_out = mido.open_output('loopMIDI Port 1')

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

def idx_to_midi(i):
    return 24 + i

note_names = [
    "c1","cs1","d1","ds1","e1","f1","fs1","g1","gs1","a1","as1","b1",
    "c2","cs2","d2","ds2","e2","f2","fs2","g2","gs2","a2","as2","b2",
    "c3","cs3","d3","ds3","e3","f3","fs3","g3","gs3","a3","as3","b3",
    "c4","cs4","d4","ds4","e4","f4","fs4","g4","gs4","a4","as4","b4",
    "c5","cs5","d5","ds5","e5","f5","fs5","g5","gs5","a5","as5","b5",
    "c6"
]

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
        callback._prev_ema = 0.0
        callback._dec_count = 0.0
        callback._current_note = None  # Currently playing note
        callback._note_buffer = {}  # Track note consistency
    
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
        # Turn off any playing note when silence is detected
        if callback._current_note is not None:
            midi_out.send(mido.Message('note_off', note=idx_to_midi(callback._current_note), velocity=0))
            print(f"NOTE OFF (silence): {note_names[callback._current_note]}")
            callback._current_note = None
            callback._note_buffer = {}  # Clear buffer on silence
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

        # compute all cosine similarities in one matrix‐vector multiply
        sig_norm = signal / (np.linalg.norm(signal) + 1e-12)
        cos_sims = dictionary_unit @ sig_norm

        # pick top note by index
        candidates = top_n_idx(cos_sims, n=5)
        top_note = candidates[0]  # Just get the best match
        
        print("Top match:", top_note, note_names[top_note])
        
        # Update note buffer with consistency tracking
        if top_note in callback._note_buffer:
            callback._note_buffer[top_note] += 1
        else:
            callback._note_buffer[top_note] = 1
            
        # Clean up buffer - remove notes that aren't the current top note
        keys_to_remove = [k for k in callback._note_buffer.keys() if k != top_note]
        for k in keys_to_remove:
            del callback._note_buffer[k]
            
        # If the same note has been detected for at least 2 callbacks, play it
        if callback._note_buffer.get(top_note, 0) >= 2:
            # If we have a new note to play
            if callback._current_note != top_note:
                # First turn off current note if one is playing
                if callback._current_note is not None:
                    midi_out.send(mido.Message('note_off', note=idx_to_midi(callback._current_note), velocity=0))
                    print(f"NOTE OFF: {note_names[callback._current_note]}")
                
                # Now play the new note
                midi_out.send(mido.Message('note_on', note=idx_to_midi(top_note), velocity=64))
                print(f"NOTE ON: {note_names[top_note]} (MIDI: {idx_to_midi(top_note)})")
                callback._current_note = top_note

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