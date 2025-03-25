import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage

# Create a new MIDI file and track
mid = MidiFile()  # default ticks_per_beat is 480
track = MidiTrack()
mid.tracks.append(track)

# Optional: set the tempo (here 120 BPM)
tempo = mido.bpm2tempo(120)
track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

# Optionally, add a program change to Acoustic Grand Piano (program 0)
# (Channel numbers in mido are 0-indexed; default channel 0 is fine.)
track.append(Message('program_change', program=0, time=0))

# Define MIDI note numbers (C4 = 60, D4 = 62, E4 = 64, F4 = 65, G4 = 67, A4 = 69)
# "Twinkle Twinkle Little Star" melody in C major:
# First phrase: C C G G A A G, Second phrase: F F E E D D C.
melody = [
    60, 60, 67, 67, 69, 69, 67,
    65, 65, 64, 64, 62, 62, 60
]

# Use a quarter note duration (in ticks)
quarter_duration = mid.ticks_per_beat

# Add note on and note off messages for each note in the melody.
for note in melody:
    # Note on with velocity 64; time=0 means immediately after previous event.
    track.append(Message('note_on', note=note, velocity=64, time=0))
    # Note off after quarter note duration.
    track.append(Message('note_off', note=note, velocity=64, time=quarter_duration))

# Save the MIDI file
mid.save('twinkle.mid')
print("MIDI file 'twinkle.mid' created!")
