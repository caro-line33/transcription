def process(input_signal):
    windowed = input_signal * np.hanning(len(input_signal))
    fft_result = np.fft.rfft(windowed)
    spectrum_full = abs(fft_result)
    spectrum = spectrum_full[26:]
    return spectrum