'''
Guitar tuner script based on the Harmonic Product Spectrum (HPS)
Upgraded for real-time MIDI output with accurate note, rest, and left-hand-only filtering
Now opens two LoopMIDI ports: one for general/right-hand (`loopMIDI Port 1`) and one for left-hand (`loopMIDI Port left`).
MIT License
Copyright (c) 2021 chciken
'''

import os
import time
import argparse
import numpy as np
import scipy.fftpack
import sounddevice as sd
import rtmidi

# General settings
SAMPLE_FREQ = 48000       # sampling rate in Hz
WINDOW_SIZE = 48000       # DFT window size
WINDOW_STEP = 12000       # hop size
NUM_HPS = 5               # number of harmonics in HPS
POWER_THRESH = 1e-6       # silence threshold
WHITE_NOISE_THRESH = 0.2  # noise floor multiplier

# MIDI ports
VPORT_MAIN = "loopMIDI Port 1"      # right-hand / default
VPORT_LEFT = "loopMIDI Port left"   # left-hand

di_main = rtmidi.MidiOut()
di_left = rtmidi.MidiOut()

# MIDI note definitions
REST_MIDI = 0            # MIDI note number used to represent a rest (left-hand channel)
REST_VEL = 64            # Velocity for rest note-on/off

# Note definitions
CONCERT_PITCH = 440      # A4 frequency
ALL_NOTES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
MIDDLE_C = 60            # MIDI note number for Middle C (C4)

# Precompute window and freq
delta_freq = SAMPLE_FREQ / WINDOW_SIZE
hann_window = np.hanning(WINDOW_SIZE)

# State for durations
current_midi = None
note_start_time = None
rest_start_time = None
in_rest = True          # start assuming rest
note_durations = []     # ("rest" or midi, duration, 'L') tuples

class HPSCallback:
    def __init__(self):
        self.buffer = [None, None]
        self.window = np.zeros(WINDOW_SIZE)

    def find_closest_note(self, freq):
        idx = int(np.round(np.log2(freq/CONCERT_PITCH) * 12))
        return 69 + idx

    def __call__(self, indata, frames, t, status):
        global current_midi, note_start_time, rest_start_time, in_rest
        if status:
            print(status)
            return

        # Slide window with latest audio
        audio = indata[:,0]
        self.window = np.roll(self.window, -len(audio))
        self.window[-len(audio):] = audio
        now = time.time()

        # Silence/rest detection
        power = np.mean(self.window**2)
        if power < POWER_THRESH:
            if not in_rest:
                # end active left-hand note
                if current_midi is not None and current_midi < MIDDLE_C:
                    di_left.send_message([0x80, current_midi, 0])
                    note_durations.append((current_midi, now - note_start_time, 'L'))
                    print(f"Note {current_midi} (L) duration: {now - note_start_time:.2f}s")
                # simulate keypress '0' for rest-on on left-hand port
                di_left.send_message([0x90, REST_MIDI, REST_VEL])
                rest_start_time = now
                in_rest = True
            return

        # Exiting rest
        if in_rest:
            di_left.send_message([0x80, REST_MIDI, 0])
            rest_duration = now - rest_start_time if rest_start_time else 0
            note_durations.append(("rest", rest_duration, 'L'))
            print(f"Rest duration: {rest_duration:.2f}s")
            in_rest = False
            rest_start_time = None

        # Compute HPS spectrum
        spec = np.abs(scipy.fftpack.fft(self.window * hann_window)[:WINDOW_SIZE//2])
        spec[:int(62/delta_freq)] = 0
        bands = [50,100,200,400,800,1600,3200,6400,12800,25600]
        for i in range(len(bands)-1):
            s = int(bands[i]/delta_freq)
            e = min(int(bands[i+1]/delta_freq), len(spec))
            avg = np.sqrt(np.sum(spec[s:e]**2)/(e-s))
            spec[s:e] *= (spec[s:e] > WHITE_NOISE_THRESH * avg)

        ip = np.interp(np.arange(0, len(spec), 1/NUM_HPS), np.arange(len(spec)), spec)
        ip /= np.linalg.norm(ip)
        hps = ip.copy()
        for h in range(1, NUM_HPS):
            tmp = hps[:int(np.ceil(len(ip)/(h+1)))] * ip[::(h+1)]
            if not tmp.any(): break
            hps = tmp

        # Detect MIDI note
        peak = np.argmax(hps)
        freq = peak * (SAMPLE_FREQ/WINDOW_SIZE) / NUM_HPS
        midi_num = self.find_closest_note(freq)

        # Debounce and left-hand filter
        self.buffer.insert(0, midi_num)
        self.buffer.pop()
        if self.buffer.count(self.buffer[0]) == len(self.buffer):
            if midi_num != current_midi and midi_num < MIDDLE_C:
                # end old left-hand note
                if current_midi is not None and current_midi < MIDDLE_C:
                    di_left.send_message([0x80, current_midi, 0])
                    note_durations.append((current_midi, now - note_start_time, 'L'))
                    print(f"Note {current_midi} (L) duration: {now - note_start_time:.2f}s")
                # start new left-hand note
                di_left.send_message([0x90, midi_num, 100])
                current_midi = midi_num
                note_start_time = now

# MIDI setup
def setup_midi():
    # Open main (right-hand) port
    ports = di_main.get_ports()
    idx = next((i for i,n in enumerate(ports) if VPORT_MAIN in n), 0)
    di_main.open_port(idx)
    print("Opened main port:", ports[idx])
    # Open left-hand port
    ports_l = di_left.get_ports()
    idx_l = next((i for i,n in enumerate(ports_l) if VPORT_LEFT in n), 0)
    di_left.open_port(idx_l)
    print("Opened left-hand port:", ports_l[idx_l])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-rests', action='store_true')
    args = parser.parse_args()
    setup_midi()
    if args.test_rests:
        # test sending only rest messages on left-hand port
        for _ in range(3):
            di_left.send_message([0x90, REST_MIDI, REST_VEL])
            time.sleep(1.0)
            di_left.send_message([0x80, REST_MIDI, 0])
            time.sleep(0.2)
    else:
        cb = HPSCallback()
        try:
            with sd.InputStream(channels=1, samplerate=SAMPLE_FREQ, blocksize=WINDOW_STEP, callback=cb):
                while True: time.sleep(0.1)
        except KeyboardInterrupt:
            now = time.time()
            # finalize left-hand note
            if current_midi is not None and current_midi < MIDDLE_C:
                di_left.send_message([0x80, current_midi, 0])
                note_durations.append((current_midi, now - note_start_time, 'L'))
            # finalize rest if active
            if in_rest and rest_start_time is not None:
                di_left.send_message([0x80, REST_MIDI, 0])
                note_durations.append(("rest", now - rest_start_time, 'L'))
            print(note_durations)