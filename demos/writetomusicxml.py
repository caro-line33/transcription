from music21 import stream, note, metadata

# Create a new score
score = stream.Score()

# Add metadata for the score
score.insert(0, metadata.Metadata())
score.metadata.title = "Example Score"
score.metadata.composer = "Your Name"

# Create a part (e.g., for a single instrument)
part = stream.Part()

# Create a measure and add a note
measure1 = stream.Measure(number=1)
# Create a note: C4 lasting one quarter note (duration=1)
n = note.Note("C4", quarterLength=1)
measure1.append(n)

# Append the measure to the part, then the part to the score
part.append(measure1)
score.append(part)

# Write the score to a MusicXML file
score.write('musicxml', fp='example_score.musicxml')
