
#  note: Guitar strings are from the top EADGBE .
#        E2: 83.41Hz from 1/period duration of sinusoidal "pure tone" wave
#        A2: 110 Hz
#        D3: 146.83 Hz
#        G3: 196 Hz
#        B3: 246.94 Hz
#        E4: 329.63 Hz
# Pure tone = sound with sinusoidal wave
# We assume equal temperament = tuning system, so that when dividing an interval of notes, the frequency ratio of
# each subsequent note is the same. In Western, we use 12 equal/12 tone temperament. These ratios are taken from the
# concert pitch 440 Hz.
# Concert pitch/standard pitch = 440 Hz is the A(la) above the middle C.
# We also assume standard concert pitch 440 Hz
# pitch = the frequency we get from user
#


import numpy as np

concert_pitch = 440
all_notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def find_closest_note(pitch):
    i = int(np.round(np.log2(pitch/concert_pitch)*12))
    closest_note = all_notes[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = concert_pitch*(2**(i/12))
    return closest_note, closest_pitch


pitch_from_user = int(input("What is the pitch?"))
note, pitch = find_closest_note(pitch_from_user)
print("The note at that pitch is " + str(note))
print("The pure pitch of that note is " + str(pitch))

import time
import scipy.io.wavfile
import sounddevice as sd

sampling_frequency = 44100
sampling_duration = 2

print('Grab your instrument!')
time.sleep(1)
recording = sd.rec(sampling_frequency*sampling_duration, samplerate=sampling_frequency, channels=1, dtype='float64')

