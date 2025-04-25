import numpy as np
import os

# Chromatic order from C1 to C6
note_order = [
    "c1", "cs1", "d1", "ds1", "e1", "f1", "fs1", "g1", "gs1", "a1", "as1", "b1",
    "c2", "cs2", "d2", "ds2", "e2", "f2", "fs2", "g2", "gs2", "a2", "as2", "b2",
    "c3", "cs3", "d3", "ds3", "e3", "f3", "fs3", "g3", "gs3", "a3", "as3", "b3",
    "c4", "cs4", "d4", "ds4", "e4", "f4", "fs4", "g4", "gs4", "a4", "as4", "b4",
    "c5", "cs5", "d5", "ds5", "e5", "f5", "fs5", "g5", "gs5", "a5", "as5", "b5",
    "c6"
]

# Load each .npy note vector from current directory and stack
arrays = []
for name in note_order:
    filename = f"{name}.npy"
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Missing file: {filename}")
    arr = np.load(filename)
    arrays.append(arr)

# Combine into a 2D matrix: each row is one note vector
note_matrix = np.vstack(arrays)

# Save the combined matrix
output_path = "notes_matrix.npy"
np.save(output_path, note_matrix)
print(f"Saved combined note matrix to '{output_path}', shape = {note_matrix.shape}")
