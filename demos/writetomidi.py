from midiutil import MIDIFile
# Create a MIDIFile object with 1 track
my_midi = MIDIFile(1)
track = 0
time = 0
my_midi.addTrackName(track, time, "Sample Track")
my_midi.addTempo(track, time, 120) # Set tempo to 120 BPM
channel = 0
pitch = 60 # MIDI note number (C4)
duration = 1 # In beats
volume = 100 # 0-127
start_time = 1 # In beats

my_midi.addNote(track, channel, pitch, start_time, duration, volume)
with open("output.mid", "wb") as output_file:
    my_midi.writeFile(output_file)
