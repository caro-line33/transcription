import os
import time
import argparse
import queue
import threading

import numpy as np
import scipy.fftpack
import sounddevice as sd
import rtmidi
from tkinter import Tk, Label

# CONSTANTS
SAMPLE_FREQ       = 48000     # sampling rate in Hz
WINDOW_SIZE       = 48000     # DFT window size
WINDOW_STEP       = 12000     # hop size
NUM_HPS           = 5         # harmonics for HPS
POWER_THRESH      = 1e-6      # silence threshold
WHITE_NOISE_THRESH= 0.2       # noise floor multiplier

VPORT_MAIN  = "loopMIDI Port 1"    # right-hand
VPORT_LEFT  = "loopMIDI Port left" # left-hand

CONCERT_PITCH = 440                # A4
ALL_NOTES     = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
MIDDLE_C      = 60                 # MIDI note number for C4

# original frequency bin width & hanning window
delta_freq = SAMPLE_FREQ / WINDOW_SIZE
hann_window = np.hanning(WINDOW_SIZE)

# midi outputs
di_main = rtmidi.MidiOut()
di_left = rtmidi.MidiOut()

note_queue = queue.Queue()

class HPSCallback:
    def __init__(self):
        self.history = [None, None]
        self.window  = np.zeros(WINDOW_SIZE)
        self.current = None

    def find_closest_note(self, freq):
        idx = int(np.round(np.log2(freq/CONCERT_PITCH) * 12))
        return 69 + idx

    def __call__(self, indata, frames, t, status):
        if status:
            return
        audio = indata[:,0]
        self.window = np.roll(self.window, -len(audio))
        self.window[-len(audio):] = audio

        # silence check
        if np.mean(self.window**2) < POWER_THRESH:
            return

        # adjust values according to frequency bands
        spec = np.abs(scipy.fftpack.fft(self.window * hann_window)[:WINDOW_SIZE//2])
        spec[:int(62/delta_freq)] = 0
        bands = [50,100,200,400,800,1600,3200,6400,12800,25600]
        for i in range(len(bands)-1):
            s = int(bands[i]/delta_freq)
            e = min(int(bands[i+1]/delta_freq), len(spec))
            avg = np.sqrt(np.sum(spec[s:e]**2)/(e-s))
            spec[s:e] *= (spec[s:e] > WHITE_NOISE_THRESH * avg)

        ip = np.interp(np.arange(0, len(spec), 1/NUM_HPS),
                       np.arange(len(spec)), spec)
        ip /= np.linalg.norm(ip)
        hps = ip.copy()
        for h in range(1, NUM_HPS):
            tmp = hps[:int(np.ceil(len(ip)/(h+1)))] * ip[::(h+1)]
            if not tmp.any(): break
            hps = tmp

        # detect peak, find note value
        peak = np.argmax(hps)
        freq = peak * (SAMPLE_FREQ/WINDOW_SIZE) / NUM_HPS
        midi = self.find_closest_note(freq)

        self.history.insert(0, midi)
        self.history.pop()
        if self.history.count(self.history[0]) == len(self.history):
            if midi != self.current:
                self.current = midi
                hand = 'L' if midi < MIDDLE_C else 'R'
                # send MIDI
                port = di_left if hand=='L' else di_main
                # note-off previous
                if hasattr(self, 'prev') and self.prev is not None:
                    port.send_message([0x80, self.prev, 0])
                # note-on new
                port.send_message([0x90, midi, 100])
                self.prev = midi

                # enqueue for GUI
                note_queue.put((midi, hand))

def setup_midi():
    # open main port
    ports = di_main.get_ports()
    i = next((i for i,n in enumerate(ports) if VPORT_MAIN in n), 0)
    di_main.open_port(i)
    print("Main port:", ports[i])
    # open left port
    ports_l = di_left.get_ports()
    j = next((j for j,n in enumerate(ports_l) if VPORT_LEFT in n), 0)
    di_left.open_port(j)
    print("Left port:", ports_l[j])

def run_gui():
    root = Tk()
    root.title("Single Note Detector")
    label = Label(root, text="Current Note:", font=("Helvetica", 32))
    label.pack(padx=20, pady=20)

    def poll():
        try:
            midi, hand = note_queue.get_nowait()
            name = ALL_NOTES[midi % 12] + str((midi//12)-1)
            label.config(text=f"{hand}-hand: {name}")
        except queue.Empty:
            pass
        root.after(50, poll)

    root.after(50, poll)
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Note Detector")
    parser.add_argument('--test-ports', action='store_true',
                        help="just open ports and exit")
    args = parser.parse_args()

    setup_midi()

    if args.test_ports:
        print("Ports opened; exiting.")
        exit(0)

    # start audio stream (runs its own thread)
    cb = HPSCallback()
    stream = sd.InputStream(channels=1,
                            samplerate=SAMPLE_FREQ,
                            blocksize=WINDOW_STEP,
                            callback=cb)
    stream.start()

    # now run GUI in the main thread
    run_gui()

    # when GUI closes, stop audio
    stream.stop()
    stream.close()
