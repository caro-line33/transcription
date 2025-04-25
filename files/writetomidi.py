import numpy as np
import sounddevice as sd
import time
import os
from scipy.optimize import nnls
import scipy.signal as signal
import rtmidi

# Constants
SAMPLE_RATE = 48000
FFT_SIZE = 48000
FREQS = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
RECORD_SECONDS = 3.0
NUM_NOTES = 88
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# MIDI setup
midi_out = rtmidi.MidiOut()
virtual_port_name = "loopMIDI Port 1"

def setup_midi():
    """Open an existing LoopMIDI port rather than trying to create one."""
    port_names = midi_out.get_ports()
    print("Available MIDI ports:", port_names)
    if not port_names:
        print("No MIDI ports found. Make sure LoopMIDI is running and you’ve added at least one port.")
        return False

    # try to find the port by name
    try:
        port_index = next(i for i, name in enumerate(port_names)
                          if virtual_port_name in name)
    except StopIteration:
        # fallback: just open the first port
        port_index = 0
        print(f"Couldn’t find a port named '{virtual_port_name}', opening '{port_names[0]}' instead.")

    try:
        midi_out.open_port(port_index)
        print(f"Connected to MIDI port: {port_names[port_index]}")
        return True
    except Exception as e:
        print(f"Failed to open MIDI port {port_names[port_index]}: {e}")
        return False


def send_midi_note(midi_note, velocity=100, duration=None):
    """
    Send MIDI note on message, and optionally a note off after specified duration
    
    Parameters:
    - midi_note: MIDI note number (0-127)
    - velocity: Note velocity (0-127)
    - duration: If provided, how long to hold the note before sending note off (in seconds)
    """
    # Note on message: [status_byte, note, velocity]
    # Status byte for note on is 0x90 + channel number (0-15)
    note_on = [0x90, midi_note, velocity]
    midi_out.send_message(note_on)
    
    # If duration is provided, send note off after waiting
    if duration is not None:
        time.sleep(duration)
        # Note off: [status_byte, note, velocity]
        # Status byte for note off is 0x80 + channel number
        note_off = [0x80, midi_note, 0]
        midi_out.send_message(note_off)

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

def record_audio(duration=RECORD_SECONDS):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return audio.flatten()

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector

def find_spectral_peaks(spectrum, freqs, min_height=None, distance=None):
    """Find significant peaks in the spectrum using scipy's peak finder"""
    if min_height is None:
        min_height = 0.05 * np.max(spectrum)
    
    if distance is None:
        # Default to minimum distance of 10 Hz between peaks
        distance = int(10 / (freqs[1] - freqs[0]))
    
    peak_indices, _ = signal.find_peaks(spectrum, height=min_height, distance=distance)
    
    peaks = []
    for idx in peak_indices:
        if idx < len(freqs):
            peaks.append((freqs[idx], spectrum[idx], idx))
    
    # Sort by amplitude (descending)
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    return peaks

def find_feasible_notes_advanced(spectrum, freqs):
    """
    A more robust approach to find feasible notes based on spectral peaks
    and harmonic relationships.
    """
    # Find significant peaks in the spectrum
    peaks = find_spectral_peaks(spectrum, freqs)
    
    if not peaks:
        print("No significant peaks found in the spectrum")
        return []
    
    print(f"\nFound {len(peaks)} significant peaks in the spectrum")
    print("Top 5 peaks:")
    for i, (freq, amp, _) in enumerate(peaks[:5]):
        print(f"  Peak {i+1}: {freq:.1f} Hz, Amplitude: {amp:.6f}")
    
    # Get note frequencies
    note_freqs = np.array([note_to_hz(i + 21) for i in range(NUM_NOTES)])
    
    # Check which notes have peaks near their fundamental and 2nd harmonic
    feasible_notes = []
    
    print("\n🔍 Checking notes for harmonic matches...")
    
    for note_idx in range(NUM_NOTES):
        f0 = note_freqs[note_idx]
        
        # Skip very low notes (below 30 Hz) - they're often problematic
        if f0 < 30:
            continue
            
        # Define frequency ranges around fundamental and harmonic (±4%)
        f0_min, f0_max = f0 * 0.96, f0 * 1.04
        f1_min, f1_max = f0 * 1.96, f0 * 2.04  # 2nd harmonic
        
        # Check if any peaks fall within these ranges
        f0_peak = None
        f1_peak = None
        
        for freq, amp, idx in peaks:
            if f0_min <= freq <= f0_max and (f0_peak is None or amp > f0_peak[1]):
                f0_peak = (freq, amp, idx)
            elif f1_min <= freq <= f1_max and (f1_peak is None or amp > f1_peak[1]):
                f1_peak = (freq, amp, idx)
        
        # A note is feasible if both fundamental and 2nd harmonic have peaks
        if f0_peak is not None and f1_peak is not None:
            midi_note = note_idx + 21
            octave = (midi_note - 12) // 12
            note_index = (midi_note - 12) % 12
            note_name = f"{NOTE_NAMES[note_index]}{octave}"
            
            # Calculate harmonic ratio (should be close to 2.0 for ideal harmonics)
            harmonic_ratio = f1_peak[0] / f0_peak[0]
            
            print(f"✓ {note_name} ({f0:.1f} Hz) - Found f0={f0_peak[0]:.1f} Hz and 2f0={f1_peak[0]:.1f} Hz, ratio={harmonic_ratio:.2f}")
            feasible_notes.append(note_idx)
    
    if not feasible_notes:
        print("No notes with both fundamental and harmonic peaks found.")
    else:
        print(f"\nFound {len(feasible_notes)} feasible notes with both f0 and 2f0 peaks.")
    
    return feasible_notes

def enhanced_note_detection(audio, dictionary, edges):
    """
    Enhanced note detection with improved spectral analysis.
    """
    # Apply Hann window and compute FFT
    windowed = audio * np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(windowed, n=FFT_SIZE))
    
    # Frequency array
    freqs = FREQS
    
    # Find feasible notes based on spectral peaks and harmonics
    feasible_notes = find_feasible_notes_advanced(spectrum, freqs)
    
    if not feasible_notes:
        print("No feasible notes detected.")
        return []
    
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
        coefficients[coefficients < 0.05 * coef_max] = 0  # 5% threshold
    
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
    
    return detected_notes

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
    
    # Create bin edges
    edges = make_custom_bins(FREQS)
    
    print("Enhanced Piano Note Detection with MIDI Output")
    print("============================================")
    
    # Setup MIDI
    if not setup_midi():
        print("Failed to set up MIDI. Exiting.")
        return
    
    print(f"\nVirtual MIDI port '{virtual_port_name}' is ready!")
    print("You can now connect to this port from MuseScore or other MIDI software.")
    
    # Main loop
    try:
        while True:
            print("\nOptions:")
            print("1. Record and analyze with enhanced detection")
            print("2. Record and analyze with MIDI output")
            print("3. Test MIDI connection with C major chord")
            print("4. Exit")
            choice = input("Choice: ")
            
            if choice == "4":
                break
                
            if choice == "1":
                try:
                    audio = record_audio()
                    
                    start_time = time.time()
                    detected_notes = enhanced_note_detection(audio, dictionary, edges)
                    elapsed = time.time() - start_time
                    
                    print(f"\nDetected {len(detected_notes)} notes in {elapsed:.3f} seconds:")
                    
                    for i, (note_name, midi_note, freq, confidence) in enumerate(detected_notes):
                        print(f"{i+1}. {note_name} (MIDI: {midi_note}, {freq:.1f} Hz) - Confidence: {confidence:.3f}")
                except Exception as e:
                    print(f"Error during analysis: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif choice == "2":
                try:
                    audio = record_audio()
                    
                    start_time = time.time()
                    detected_notes = enhanced_note_detection(audio, dictionary, edges)
                    elapsed = time.time() - start_time
                    
                    print(f"\nDetected {len(detected_notes)} notes in {elapsed:.3f} seconds:")
                    
                    # Play the detected notes via MIDI
                    print("\nSending detected notes to MIDI output...")
                    
                    # First, calculate velocity based on confidence
                    for i, (note_name, midi_note, freq, confidence) in enumerate(detected_notes):
                        # Scale confidence to MIDI velocity (0-127)
                        velocity = int(confidence * 100) + 27  # minimum velocity of 27, max of 127
                        velocity = min(127, max(1, velocity))  # ensure in valid range
                        
                        print(f"  → {note_name} (MIDI: {midi_note}) - Velocity: {velocity}")
                        send_midi_note(midi_note, velocity)
                    
                    # Hold notes for a moment
                    time.sleep(1.0)
                    
                    # Send note-off for all notes
                    for _, midi_note, _, _ in detected_notes:
                        midi_out.send_message([0x80, midi_note, 0])
                        
                except Exception as e:
                    print(f"Error during analysis or MIDI output: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif choice == "3":
                print("Testing MIDI connection with C major chord...")
                # Play C major chord (C4, E4, G4)
                send_midi_note(60, 100)  # C4
                send_midi_note(64, 100)  # E4
                send_midi_note(67, 100)  # G4
                
                # Hold for 1 second
                time.sleep(1.0)
                
                # Turn off notes
                midi_out.send_message([0x80, 60, 0])  # C4 off
                midi_out.send_message([0x80, 64, 0])  # E4 off
                midi_out.send_message([0x80, 67, 0])  # G4 off
                
                print("Test complete. Did you hear the chord in MuseScore?")
            
            print()
    
    finally:
        # Clean up MIDI port
        print("\nClosing MIDI port...")
        midi_out.close_port()

if __name__ == "__main__":
    main()