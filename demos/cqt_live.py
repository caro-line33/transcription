import numpy as np
import sounddevice as sd
import time
import os
import threading
import scipy.signal as signal
from scipy.optimize import nnls
from collections import deque

# Constants
SAMPLE_RATE = 48000
FFT_SIZE = 48000
BUFFER_SECONDS = 1.5  # Reduced buffer size for faster response
HOP_SECONDS = 0.25    # Shorter hop for more frequent updates
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_SECONDS)
HOP_SIZE = int(SAMPLE_RATE * HOP_SECONDS)
NUM_NOTES = 88
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Silence detection thresholds
TIME_DOMAIN_SILENCE_THRESHOLD = 1e-3
FREQ_DOMAIN_SILENCE_THRESHOLD = 1e-3

# Note detection parameters
HARMONIC_TOLERANCE = 0.04
MIN_PEAK_HEIGHT = 0.05
PEAK_DISTANCE_HZ = 10

# Note stability tracking
NOTE_STABILITY_THRESHOLD = 3  # Number of consecutive frames to consider a note stable

# Global variables
audio_buffer = deque(maxlen=BUFFER_SIZE)
buffer_lock = threading.Lock()
active_notes = []
is_running = False
note_history = []  # Track note changes over time
history_length = 4  # Number of frames to keep in history
note_stability_counters = {}  # Track consecutive detections for each note
was_silent_before = True  # Track transitions from silence to sound

def note_to_hz(midi_note):
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def make_custom_bins(freqs):
    bin_plan = [(100, 1), (400, 2), (900, 3), (1600, 4), (2500, 5), (3600, 6)]
    edges = [freqs[0]]
    idx = 0
    
    for width, bin_width in bin_plan:
        for _ in range(width // bin_width):
            next_idx = min(idx + bin_width, len(freqs) - 1)
            next_edge = freqs[next_idx]
            edges.append(next_edge)
            idx = next_idx
    
    if edges[-1] < freqs[-1]:
        edges.append(freqs[-1])
    
    return np.array(edges)

def compress_fft_to_custom_bins(spectrum, freqs, edges, threshold=1e-5):
    compressed = []
    spectrum = np.copy(spectrum)
    spectrum[spectrum < threshold] = 0
    
    for i in range(len(edges) - 1):
        start = np.searchsorted(freqs, edges[i], side='left')
        end = np.searchsorted(freqs, edges[i + 1], side='right')
        compressed.append(np.sum(spectrum[start:end]))
    
    return np.array(compressed)

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector

def is_silent(audio_data, spectrum=None):
    """Check if audio is silent using both time and frequency domain checks."""
    # Time domain check
    rms = np.sqrt(np.mean(audio_data**2))
    if rms < TIME_DOMAIN_SILENCE_THRESHOLD:
        return True, f"Time-domain silence (RMS: {rms:.8f})"
    
    # Frequency domain check (if spectrum provided)
    if spectrum is not None:
        # Filter out very small values
        nonzero_spectrum = spectrum[spectrum > 1e-6]
        
        if len(nonzero_spectrum) == 0:
            return True, "No significant frequency components"
        
        # Check average energy of non-zero frequencies
        avg_energy = np.mean(nonzero_spectrum)
        if avg_energy < FREQ_DOMAIN_SILENCE_THRESHOLD:
            return True, f"Frequency-domain silence (Avg energy: {avg_energy:.8f})"
    
    return False, "Audio contains signal"

def find_spectral_peaks(spectrum, freqs, min_height=None, distance=None):
    """Find significant peaks in the spectrum using scipy's peak finder"""
    if min_height is None:
        min_height = MIN_PEAK_HEIGHT * np.max(spectrum)
    
    if distance is None:
        # Convert Hz to bin distance
        distance = int(PEAK_DISTANCE_HZ / (freqs[1] - freqs[0]))
    
    peak_indices, peak_props = signal.find_peaks(
        spectrum, 
        height=min_height, 
        distance=distance,
        prominence=min_height * 0.5
    )
    
    peaks = []
    for idx in peak_indices:
        if idx < len(freqs):
            peaks.append((freqs[idx], spectrum[idx], idx))
    
    # Sort by amplitude (descending)
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    return peaks

def is_harmonic_of_stronger_fundamental(freq, amplitude, all_peaks, harmonic_tolerance=0.02):
    """Check if a frequency is likely a harmonic of a stronger fundamental."""
    # Only consider peaks with higher amplitude than this one
    stronger_peaks = [p for p in all_peaks if p[1] > amplitude]
    
    for h in range(2, 8):  # Check if it's the 2nd-7th harmonic of any stronger peak
        for fundamental_freq, fund_amp, _ in stronger_peaks:
            expected_harmonic = fundamental_freq * h
            # If this frequency is within tolerance of being the h-th harmonic of a stronger peak
            if abs(freq - expected_harmonic) / expected_harmonic < harmonic_tolerance:
                return True
    
    return False

def handle_silence_to_sound_transition(spectrum, freqs):
    """
    Special handling for transitions from silence to sound.
    More conservative detection to avoid false positives.
    """
    # Use higher peak threshold for transition frames
    peaks = find_spectral_peaks(spectrum, freqs, min_height=0.2 * np.max(spectrum))
    
    if not peaks:
        return []
    
    # Only consider the top 3 strongest peaks
    top_peaks = peaks[:min(3, len(peaks))]
    
    # Get note frequencies
    note_freqs = np.array([note_to_hz(i + 21) for i in range(NUM_NOTES)])
    
    # Find closest notes to these peaks
    transition_notes = []
    
    for freq, amp, _ in top_peaks:
        # Find closest note
        note_idx = np.argmin(np.abs(note_freqs - freq))
        
        # Skip very low notes (below 30 Hz)
        if note_freqs[note_idx] < 30:
            continue
            
        # Add to transition notes if not already there
        if note_idx not in transition_notes:
            transition_notes.append(note_idx)
    
    return transition_notes

def find_feasible_notes_advanced(spectrum, freqs, is_transition=False):
    """
    Find feasible notes based on spectral peaks and harmonic relationships,
    with improved harmonic disambiguation.
    """
    # Special handling for silence to sound transitions
    if is_transition:
        return handle_silence_to_sound_transition(spectrum, freqs)
    
    # Find significant peaks in the spectrum
    peaks = find_spectral_peaks(spectrum, freqs)
    
    if not peaks:
        return []
    
    # Get note frequencies for all 88 piano notes
    note_freqs = np.array([note_to_hz(i + 21) for i in range(NUM_NOTES)])
    
    # Filter out peaks that are likely harmonics of stronger fundamentals
    filtered_peaks = []
    for freq, amp, idx in peaks:
        if not is_harmonic_of_stronger_fundamental(freq, amp, peaks, HARMONIC_TOLERANCE):
            filtered_peaks.append((freq, amp, idx))
    
    # Check which notes have peaks near their fundamental and 2nd harmonic
    feasible_notes = []
    
    for note_idx in range(NUM_NOTES):
        f0 = note_freqs[note_idx]
        
        # Skip very low notes (below 30 Hz)
        if f0 < 30:
            continue
            
        # Define frequency ranges around fundamental and harmonic
        f0_min, f0_max = f0 * (1 - HARMONIC_TOLERANCE), f0 * (1 + HARMONIC_TOLERANCE)
        f1_min, f1_max = f0 * 2 * (1 - HARMONIC_TOLERANCE), f0 * 2 * (1 + HARMONIC_TOLERANCE)
        
        # Check if any peaks fall within these ranges
        f0_peak = None
        f1_peak = None
        
        # First check filtered peaks (likely fundamentals) for f0
        for freq, amp, idx in filtered_peaks:
            if f0_min <= freq <= f0_max and (f0_peak is None or amp > f0_peak[1]):
                f0_peak = (freq, amp, idx)
        
        # For f1 (2nd harmonic), check all peaks
        for freq, amp, idx in peaks:
            if f1_min <= freq <= f1_max and (f1_peak is None or amp > f1_peak[1]):
                f1_peak = (freq, amp, idx)
        
        # A note is feasible if both fundamental and 2nd harmonic have peaks
        if f0_peak is not None and f1_peak is not None:
            feasible_notes.append(note_idx)
    
    return feasible_notes

def track_note_stability(detected_notes):
    """
    Track note stability across frames.
    Returns a set of stable notes (detected for multiple consecutive frames).
    """
    global note_stability_counters
    
    # Current detected MIDI notes
    current_midi_notes = {midi for _, midi, _, _ in detected_notes}
    
    # Update counters
    for midi in list(note_stability_counters.keys()):
        if midi in current_midi_notes:
            # Increment counter for detected notes
            note_stability_counters[midi] += 1
        else:
            # Remove counter for notes no longer detected
            note_stability_counters.pop(midi)
    
    # Add new notes
    for _, midi, _, _ in detected_notes:
        if midi not in note_stability_counters:
            note_stability_counters[midi] = 1
    
    # Return stable notes (detected for multiple consecutive frames)
    stable_notes = {midi for midi, count in note_stability_counters.items() 
                   if count >= NOTE_STABILITY_THRESHOLD}
    
    return stable_notes

def enhanced_note_detection(audio_data, dictionary, edges, is_transition=False):
    """Enhanced note detection with improved harmonic disambiguation."""
    # Check for silence in time domain first (quick check)
    silent, message = is_silent(audio_data)
    if silent:
        return [], message
    
    # Apply Hann window and compute FFT
    windowed = audio_data * np.hanning(len(audio_data))
    spectrum = np.abs(np.fft.rfft(windowed, n=FFT_SIZE))
    
    # Check for silence in frequency domain
    silent, message = is_silent(audio_data, spectrum)
    if silent:
        return [], message
    
    # Frequency array
    freqs = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
    
    # Find feasible notes with improved harmonic discrimination
    feasible_notes = find_feasible_notes_advanced(spectrum, freqs, is_transition)
    
    if not feasible_notes:
        return [], "No feasible notes detected"
    
    # Compress spectrum for NNLS
    compressed = compress_fft_to_custom_bins(spectrum, freqs, edges)
    
    # Normalize compressed spectrum
    compressed_normalized = normalize_vector(compressed)
    
    # Use reduced dictionary with only feasible notes
    D_reduced = dictionary[:, feasible_notes]
    
    # Solve NNLS
    coefficients, _ = nnls(D_reduced, compressed_normalized)
    
    # Apply threshold
    coef_max = np.max(coefficients)
    if coef_max > 0:
        # Adaptive thresholding based on note position
        for i, coef in enumerate(coefficients):
            note_idx = feasible_notes[i]
            
            # Higher threshold for transition frames
            if is_transition:
                threshold_factor = 0.10  # Higher threshold during transitions
            else:
                # Higher threshold for higher notes
                threshold_factor = 0.05 + 0.03 * (note_idx / NUM_NOTES)
                
            if coef < threshold_factor * coef_max:
                coefficients[i] = 0
    
    # Get note information for detected notes
    note_freqs = np.array([note_to_hz(i + 21) for i in range(NUM_NOTES)])
    detected_notes = []
    
    for i, coef in enumerate(coefficients):
        if coef > 0:
            note_idx = feasible_notes[i]
            midi_note = note_idx + 21
            octave = (midi_note - 12) // 12
            note_index = (midi_note - 12) % 12
            note_name = f"{NOTE_NAMES[note_index]}{octave}"
            freq = note_freqs[note_idx]
            confidence = coef / coef_max
            
            detected_notes.append((note_name, midi_note, freq, confidence))
    
    return detected_notes, "Notes detected"

def filter_by_stability(detected_notes):
    """Filter notes based on stability tracking."""
    stable_midi_notes = track_note_stability(detected_notes)
    
    # Keep only stable notes or previously stable notes with high confidence
    filtered_notes = []
    for note_name, midi, freq, conf in detected_notes:
        if midi in stable_midi_notes or (midi in note_stability_counters and conf > 0.7):
            filtered_notes.append((note_name, midi, freq, conf))
    
    return filtered_notes

def audio_callback(indata, frames, time, status):
    """Callback function for audio input stream."""
    if status:
        print(f"Audio status: {status}")
    
    # Add new audio data to buffer
    with buffer_lock:
        audio_buffer.extend(indata[:, 0])

def analysis_thread(dictionary, edges):
    """Thread function for continuous analysis."""
    global active_notes, is_running, note_history, was_silent_before
    
    # Sleep initially to allow buffer to fill
    time.sleep(BUFFER_SECONDS)
    
    last_analysis_time = time.time()
    silence_message = "Awaiting audio..."
    
    while is_running:
        current_time = time.time()
        
        # Only analyze at each hop time
        if current_time - last_analysis_time >= HOP_SECONDS:
            # Get a copy of the current buffer for analysis
            with buffer_lock:
                if len(audio_buffer) < BUFFER_SIZE:
                    continue  # Not enough data yet
                buffer_copy = np.array(list(audio_buffer))
            
            # Check for silence (quick check)
            is_silent_now, _ = is_silent(buffer_copy)
            
            # Detect silence-to-sound transition
            is_transition = was_silent_before and not is_silent_now
            if is_transition:
                print("⚡ Detected transition from silence to sound")
            
            # Run note detection with silence checking
            detected, message = enhanced_note_detection(buffer_copy, dictionary, edges, is_transition)
            
            # Update silence state
            was_silent_before = is_silent_now
            
            # Update note history and apply stability filtering
            if detected:
                # Add to history
                note_history.append(detected)
                if len(note_history) > history_length:
                    note_history.pop(0)
                
                # Apply stability filtering
                filtered_notes = filter_by_stability(detected)
                
                active_notes = filtered_notes
                silence_message = message
            else:
                # Reset stability counters and history if no notes detected
                if is_silent_now:
                    note_stability_counters.clear()
                    note_history = []
                
                active_notes = []
                silence_message = message
            
            # Clear terminal and print current status
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n🎹 Real-time Piano Note Detection")
            print("===============================")
            
            if not active_notes:
                print(f"\n  Status: {silence_message}")
            else:
                print("\n  Currently Detected Notes:")
                
                # Print stability counters alongside notes
                stable_note_ids = {midi for midi, count in note_stability_counters.items() 
                                 if count >= NOTE_STABILITY_THRESHOLD}
                
                # Sort by MIDI number for consistent display
                sorted_notes = sorted(active_notes, key=lambda x: x[1])
                for i, (note_name, midi_note, freq, confidence) in enumerate(sorted_notes):
                    # Show stability indicator
                    stability = note_stability_counters.get(midi_note, 0)
                    stability_indicator = "✓" if midi_note in stable_note_ids else str(stability)
                    
                    print(f"    {i+1}. {note_name} (MIDI: {midi_note}, {freq:.1f} Hz) - " +
                          f"Confidence: {confidence:.3f} [Stability: {stability_indicator}]")
            
            print("\nBuffer: {:.1f}s | Hop: {:.1f}s | Press Ctrl+C to stop".format(
                BUFFER_SECONDS, HOP_SECONDS))
            
            last_analysis_time = current_time
        
        # Sleep a short time to avoid consuming too much CPU
        time.sleep(0.01)

def start_real_time_detection(dictionary):
    """Start real-time piano note detection."""
    global audio_buffer, is_running, active_notes, note_history, note_stability_counters, was_silent_before
    
    # Reset global variables
    audio_buffer.clear()
    active_notes = []
    note_history = []
    note_stability_counters = {}
    was_silent_before = True
    is_running = True
    
    # Create bin edges
    freqs = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
    edges = make_custom_bins(freqs)
    
    # Start analysis thread
    analysis_thread_handle = threading.Thread(target=analysis_thread, args=(dictionary, edges))
    analysis_thread_handle.daemon = True
    analysis_thread_handle.start()
    
    # Start audio input stream
    stream = sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        callback=audio_callback,
        blocksize=1024
    )
    
    print("Starting real-time piano note detection...")
    print("Buffer size: {:.1f} seconds".format(BUFFER_SECONDS))
    print("Analysis hop time: {:.1f} seconds".format(HOP_SECONDS))
    print("Press Ctrl+C to stop")
    
    try:
        with stream:
            while True:
                time.sleep(0.1)  # Keep main thread alive
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        is_running = False
        if analysis_thread_handle.is_alive():
            analysis_thread_handle.join(timeout=1.0)
        print("Real-time detection stopped")

def main():
    # Try to load the dictionary
    dictionary_path = "recorded_note_dictionary_normalized.npy"
    
    if not os.path.exists(dictionary_path):
        print(f"Error: Dictionary file '{dictionary_path}' not found.")
        print("Please run the dictionary builder script first.")
        return
    
    try:
        dictionary = np.load(dictionary_path)
        print(f"Loaded dictionary with shape: {dictionary.shape}")
    except Exception as e:
        print(f"Error loading dictionary: {e}")
        return
    
    print("Real-Time Piano Note Detection")
    print("=============================")
    
    try:
        start_real_time_detection(dictionary)
    except Exception as e:
        print(f"Error during real-time detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()