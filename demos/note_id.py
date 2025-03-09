''' after finding the fundamental frequency of a sample we assign it a musical note.
there is a mathematical formula for finding the frequency of a note which is i half-steps away from another
note. we can use the inverse of this to calculate the approximate number of half steps between a detected 
frequency, and a certain reference frequency. from this we can assign a note and octave to the detected
frequency, based on its distance from the reference frequency. '''

import numpy as np

detected_frequency = float(input("what is the frequency?" ))

a0 = 27.5 # lowest note on piano
notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"] # all notes

number_of_steps = int(np.round((np.log2(detected_frequency/a0))*12)) 
    # inverse formula for half steps
octave = (9+number_of_steps)//12 
    # new octave starts at C, add 9 steps to shift start to A, our reference
note = notes[number_of_steps%12]
    # take the remainder of division to get # of notes from A
print(note, octave)


