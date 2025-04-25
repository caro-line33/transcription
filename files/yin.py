import sounddevice as sd
import numpy as np
import time
import os
import librosa

sample_freq = 8400    # sampling frequency
fft_length = 8400     # how many samples to buffer
threshold = 1e-3      # power threshold for pre-processing
blocksize = 1024      # reduced from 800 to help prevent overflows


# YIN is a time-domain processing technique that evaluates the periodicity of signals by looking for repetition.
def yin_tonal(frame, fs, threshold=0.10):
    N = len(frame)
    max_lag = N // 3
    
    # 1) difference function (optimized)
    d = np.zeros(max_lag)
    for τ in range(1, max_lag):
        diff = frame[:-τ] - frame[τ:]
        d[τ] = np.sum(diff * diff)
    
    # 2) cumulative mean normalization
    cumsum = np.cumsum(d[1:])
    τs = np.arange(1, max_lag)
    d_prime = np.empty_like(d)
    d_prime[0] = 1.0
    d_prime[1:] = d[1:] * τs / (cumsum + 1e-12)
    
    # 3) find first dip under threshold
    candidates = np.where(d_prime[1:] < threshold)[0]
    if candidates.size > 0:
        τ_star = candidates[0] + 1
        # 4) compute pitch
        pitch = fs / τ_star
        strength = d_prime[τ_star]
        return True, pitch, strength
    else:
        # no clear periodicity
        return False, None, np.min(d_prime[1:])

def callback(indata, frames, time_info, status):
    if status:
        print(status)
        return
    
    # one-time init of the circular buffer and state tracking
    if not hasattr(callback, "_buffer"):
        callback._buffer = np.zeros(fft_length, dtype=float)
        callback._write_idx = 0
        callback._last_tonal = False
        callback._stability = 0
        callback._history = []
    
    # Get buffer and current write index
    buf = callback._buffer
    idx = callback._write_idx
    audio = indata[:, 0]
    n = len(audio)
    
    # write new block into circular buffer
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
        # Reset tracking when silent
        callback._last_tonal = False
        callback._stability = 0
        callback._history = []

    else:
        # Process a smaller chunk to increase YIN performance
        frame_size = 2048  # Smaller than the full buffer
        current_frame = buf[-frame_size:].copy()  # Use most recent samples
        
        # Apply a simple pre-emphasis filter to reduce low frequency noise
        pre_emphasis = 0.97
        emphasized = np.append(current_frame[0], current_frame[1:] - pre_emphasis * current_frame[:-1])
        
        # Run YIN on the filtered frame
        is_tonal, freq, tonic_strength = yin_tonal(emphasized, sample_freq, threshold=0.15)
        
        # Store result in history
        callback._history.append((is_tonal, freq, tonic_strength))
        if len(callback._history) > 5:  # Keep last 5 frames
            callback._history.pop(0)
        
        # Calculate tonal stability
        tonal_count = sum(item[0] for item in callback._history)
        tonal_ratio = tonal_count / max(1, len(callback._history))
        
        # Update stability with some inertia
        callback._stability = 0.7 * callback._stability + 0.3 * tonal_ratio
        
        # Create a visual representation of stability
        stability_bar = "#" * int(callback._stability * 20)
        print(f"Tonal stability: [{stability_bar:<20}] {callback._stability:.2f}")
        
        # Apply hysteresis for state changes
        current_tonal = callback._last_tonal
        if callback._stability > 0.6 and not current_tonal:
            current_tonal = True  # Change to tonal
        elif callback._stability < 0.4 and current_tonal:
            current_tonal = False  # Change to non-tonal
        
        callback._last_tonal = current_tonal
        
        if not current_tonal:
            print(f"non-tonal frame (d' min ≃ {tonic_strength:.2f})")
        else:
            # Get average frequency from recent tonal frames
            tonal_frames = [(f, s) for t, f, s in callback._history if t]
            if tonal_frames:
                # Weight by strength (lower is stronger)
                weights = [1/(s+0.01) for _, s in tonal_frames]
                sum_weight = sum(weights)
                if sum_weight > 0:
                    avg_freq = sum(f * w / sum_weight for (f, _), w in zip(tonal_frames, weights))
                    
                    # Filter out implausible frequency jumps
                    if freq is not None:
                        # Use average if close, otherwise use detected
                        if abs(freq - avg_freq) / avg_freq < 0.2:
                            freq = avg_freq
                
                # Only show frequency if we have a reasonable value
                if freq is not None and 20 <= freq <= 4000:
                    print(f"tonal!  pitch ≃ {freq:.1f} Hz  (strength={tonic_strength:.2f})")
                    
                    # Convert to note
                    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                    note_num = int(round(12 * np.log2(freq/440) + 69)) % 12
                    octave = int(np.log2(freq/440) + 4.75)
                    print(f"Note: {notes[note_num]}{octave}")
                else:
                    print(f"tonal!  (strength={tonic_strength:.2f})")
            else:
                print(f"tonal!  pitch ≃ {freq:.1f} Hz  (strength={tonic_strength:.2f})")

if __name__ == "__main__":
    try:
        with sd.InputStream(channels=1, callback=callback, samplerate=sample_freq, blocksize=blocksize):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped")
    except Exception as e:
        print(str(e))