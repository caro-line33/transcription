# if RMS signal too low, do not analyze ✅
# else, compute Hann window, and multiply with it. ✅
# compute FFT ✅
# do spectral gating based on octave band ranges.
    # find avg amplitude of each band, divide by some value so we do not lose too much info, then
    # set amplitudes below that to be 0.
# estimate which notes might possibly be harmonics, and assign a likelihood score to them
    # go through every potential fundamental frequency range,and check if it is nonzero and its harmonics exist.
# use NNLS to calculate which notes are contained, using the vectors from our likely notes and our 
# recorded spectrum as the right hand side. we are solving for coefficient matrix.
    # enforce sparsity
# use combination of NNLS and likelihood score to determine which notes are in it
    # elbow method?
# print out notes


def make_octave_bands(start_hz=27.5, num_bands=10):
    bands = [start_hz]
    for _ in range(num_bands):
        bands.append(bands[-1] * 2)
    return bands

make_octave_bands(27.5, 9)
# ➜ [27.5, 55.0, 110.0, 220.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0, 14080.0]
