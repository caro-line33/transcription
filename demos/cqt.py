import numpy as np
import sounddevice as sd
import threading
import time
import collections
from scipy.optimize import nnls

# Constants
SAMPLE_RATE = 48000
FFT_SIZE = 48000
BUFFER_SECONDS = 1.5  # Size of circular buffer
HOP_SIZE = 0.25       # Analysis every 250ms
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_SECONDS)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_SIZE)

class PianoTranscriber:
    def __init__(self, dictionary_path="recorded_note_dictionary_normalized.npy"):
        # Load dictionary and precompute values
        self.dictionary = np.load(dictionary_path)
        self.freqs = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
        self.edges = self.make_custom_bins(self.freqs)
        self.note_freqs = np.array([self.note_to_hz(i + 21) for i in range(88)])
        
        # Buffer for audio
        self.buffer = collections.deque(maxlen=BUFFER_SIZE)
        
        # State tracking
        self.note_states = [{'active': False, 'count': 0, 'confidence': 0} for _ in range(88)]
        self.last_detected_notes = []
        
        # Threading
        self.is_running = False
        self.lock = threading.Lock()
    
    def make_custom_bins(self, freqs):
        # Same bin creation logic as before
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
    
    def note_to_hz(self, midi_note):
        """Convert MIDI note number to frequency in Hz."""
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream - adds samples to buffer."""
        if status:
            print(f"Audio status: {status}")
        
        # Add new samples to the buffer
        self.buffer.extend(indata[:, 0])
    
    def analyze_buffer(self):
        """Analyze current buffer contents."""
        with self.lock:
            # Copy buffer to array
            if len(self.buffer) < BUFFER_SIZE:
                return []  # Not enough data yet
            
            audio = np.array(self.buffer)
        
        # Window and FFT
        windowed = audio * np.hanning(len(audio))
        spectrum = np.abs(np.fft.rfft(windowed, n=FFT_SIZE))
        compressed = self.compress_fft_to_custom_bins(spectrum)
        
        # Normalize
        norm = np.linalg.norm(compressed)
        if norm > 0:
            compressed_normalized = compressed / norm
        else:
            return []  # Empty spectrum
        
        # Find feasible notes
        feasible_notes, likelihood_scores = self.find_feasible_notes(compressed)
        
        if not feasible_notes:
            return []
        
        # Create reduced dictionary
        D_reduced = self.dictionary[:, feasible_notes]
        
        # Solve reduced NNLS
        coefficients, _ = nnls(D_reduced, compressed_normalized)
        
        # Apply sparsity
        coef_max = np.max(coefficients)
        if coef_max > 0:
            coefficients[coefficients < 0.08 * coef_max] = 0
        
        # Combine scores
        combined_scores = coefficients * likelihood_scores[feasible_notes]
        
        # Get detected notes
        detected_indices = np.argsort(combined_scores)[::-1][:10]  # Top 10 notes
        detected_notes = []
        
        for i in detected_indices:
            if combined_scores[i] > 0:
                note_idx = feasible_notes[i]
                detected_notes.append((note_idx, combined_scores[i]))
        
        # Update note states with hysteresis
        self.update_note_states(detected_notes)
        
        return self.get_active_notes()
    
    def compress_fft_to_custom_bins(self, spectrum, threshold=1e-6):
        """Compress FFT spectrum to custom bins."""
        compressed = []
        spectrum_copy = np.copy(spectrum)
        spectrum_copy[spectrum_copy < threshold] = 0
        
        for i in range(len(self.edges) - 1):
            start = np.searchsorted(self.freqs, self.edges[i], side='left')
            end = np.searchsorted(self.freqs, self.edges[i + 1], side='right')
            compressed.append(np.sum(spectrum_copy[start:end]))
        
        return np.array(compressed)
    
    def find_feasible_notes(self, compressed_spectrum):
        """Find notes with significant fundamental and harmonics."""
        # Similar to previous implementation
        feasible_notes = []
        likelihood_scores = np.zeros(88)
        
        # Define thresholds
        spectrum_max = np.max(compressed_spectrum)
        if spectrum_max < 1e-6:
            return [], likelihood_scores
        
        fundamental_threshold = 0.05 * spectrum_max
        harmonic_threshold = 0.03 * spectrum_max
        
        for note_idx in range(88):
            # Get fundamental bin
            fundamental_freq = self.note_freqs[note_idx]
            fundamental_bin = np.searchsorted(self.edges, fundamental_freq) - 1
            
            if fundamental_bin < 0 or fundamental_bin >= len(compressed_spectrum):
                continue
            
            # Check fundamental energy
            fundamental_energy = compressed_spectrum[fundamental_bin]
            if fundamental_energy < fundamental_threshold:
                continue
            
            # Initialize likelihood
            note_likelihood = fundamental_energy
            
            # Check harmonics
            harmonic_count = 0
            for h in range(2, 8):  # Check 7 harmonics
                harmonic_freq = fundamental_freq * h
                if harmonic_freq >= self.edges[-1]:
                    break
                
                harmonic_bin = np.searchsorted(self.edges, harmonic_freq) - 1
                if harmonic_bin >= len(compressed_spectrum):
                    break
                
                harmonic_energy = compressed_spectrum[harmonic_bin]
                if harmonic_energy > harmonic_threshold:
                    harmonic_count += 1
                    note_likelihood += harmonic_energy / h
            
            if harmonic_count >= 2:
                feasible_notes.append(note_idx)
                likelihood_scores[note_idx] = note_likelihood
        
        # Normalize
        if np.max(likelihood_scores) > 0:
            likelihood_scores = likelihood_scores / np.max(likelihood_scores)
        
        return feasible_notes, likelihood_scores
    
    def update_note_states(self, detected_notes):
        """Update note states with hysteresis to prevent flickering."""
        # First, decrease counters for all notes
        for state in self.note_states:
            if state['count'] > 0:
                state['count'] -= 1
        
        # Update detected notes
        for note_idx, confidence in detected_notes:
            self.note_states[note_idx]['count'] = min(5, self.note_states[note_idx]['count'] + 2)
            self.note_states[note_idx]['confidence'] = confidence
        
        # Update active state with hysteresis
        ON_THRESHOLD = 3   # Need 3+ counts to turn on
        OFF_THRESHOLD = 0  # Need 0 counts to turn off
        
        for i, state in enumerate(self.note_states):
            if not state['active'] and state['count'] >= ON_THRESHOLD:
                state['active'] = True
                print(f"Note ON: {self.get_note_name(i+21)}")
            elif state['active'] and state['count'] <= OFF_THRESHOLD:
                state['active'] = False
                print(f"Note OFF: {self.get_note_name(i+21)}")
    
    def get_active_notes(self):
        """Get list of currently active notes."""
        active_notes = []
        for i, state in enumerate(self.note_states):
            if state['active']:
                midi_note = i + 21
                note_name = self.get_note_name(midi_note)
                freq = self.note_freqs[i]
                confidence = state['confidence']
                active_notes.append((note_name, midi_note, freq, confidence))
        
        return active_notes
    
    def get_note_name(self, midi_note):
        """Convert MIDI note number to note name."""
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (midi_note - 12) // 12
        note_index = (midi_note - 12) % 12
        return f"{note_names[note_index]}{octave}"
    
    def analysis_thread(self):
        """Thread for continuous analysis."""
        last_analysis_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            
            # Only analyze every HOP_SIZE seconds
            if current_time - last_analysis_time >= HOP_SIZE:
                self.last_detected_notes = self.analyze_buffer()
                last_analysis_time = current_time
            
            # Sleep to reduce CPU usage
            time.sleep(0.01)
    
    def start(self):
        """Start transcription."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start analysis thread
        self.analysis_thread_handle = threading.Thread(target=self.analysis_thread)
        self.analysis_thread_handle.daemon = True
        self.analysis_thread_handle.start()
        
        # Start audio stream
        self.stream = sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            callback=self.audio_callback
        )
        self.stream.start()
        
        print("Real-time piano transcription started")
    
    def stop(self):
        """Stop transcription."""
        self.is_running = False
        
        if hasattr(self, 'stream') and self.stream.active:
            self.stream.stop()
            self.stream.close()
        
        if hasattr(self, 'analysis_thread_handle'):
            self.analysis_thread_handle.join(timeout=1.0)
        
        print("Real-time piano transcription stopped")

# Main function
def main():
    transcriber = PianoTranscriber()
    
    print("Piano Real-time Transcription")
    print("============================")
    print("Press Enter to start/stop transcription")
    print("Press 'q' to quit")
    
    is_active = False
    
    while True:
        user_input = input()
        
        if user_input.lower() == 'q':
            if is_active:
                transcriber.stop()
            break
        
        if not is_active:
            transcriber.start()
            is_active = True
        else:
            transcriber.stop()
            is_active = False
            
            # Print final notes
            print("\nDetected Notes:")
            for note_name, midi_note, freq, confidence in transcriber.last_detected_notes:
                print(f"{note_name} (MIDI: {midi_note}, {freq:.1f} Hz) - Confidence: {confidence:.3f}")

if __name__ == "__main__":
    main()






