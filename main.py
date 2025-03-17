import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import sounddevice as sd

from Biquad import BiquadFilter

def main():
    fs = 48000          # Sample rate
    duration = 1        # Duration of generated signal
    frequency = 1000   # Frequency of generated signal
    scale = 0.5     # Amplitude of generated signal

    t = np.linspace(0, duration, int(fs * duration), endpoint=False, dtype=np.float64)
    signal = np.random.normal(0, 1, len(t)) * scale

    # Define our coordinates for the poles and zeroes
    zeroes = np.array([0.65])
    poles = np.array([0.75, 0.70])
    gain = 0.215

    # Compute the transfer function and get the coefficients
    zeroCoeffs, poleCoeffs = sp.signal.zpk2tf(zeroes, poles, gain)
    # Compute frequency response
    w, h = sp.signal.freqz(zeroCoeffs, poleCoeffs, worN=1024, fs=fs)

    print(zeroCoeffs)
    print(poleCoeffs)

    # Our plot stuff is below

    # Plot the Pole-zero plot with the filter magnitude and phase plots
    figPoleZero, axesPoleZero = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

    PreparePoleZeroPlot(axesPoleZero[0], zeroes, poles, 'Pole-Zero Plot')
    PrepareFreqRespPlot(axesPoleZero[1], w, h, fs / 2, 'Magnitude Response')
    PreparePhaseRespPlot(axesPoleZero[2], w, h, fs / 2, 'Phase Response')

    plt.subplots_adjust(wspace=0.3)

    # Plot the signal and frequency spectra
    figSig, axesSig = plt.subplots(2, 2, figsize=(15, 5))  # Adjust figsize as needed

    PrepareSignalPlot(axesSig[0, 0], signal, t, fs, 0.005, 'Signal')
    PrepareFFTPlot(axesSig[0, 1], signal, t, fs, 'FFT of Signal')

    PlaySound(signal, fs)
    processedSig = sp.signal.lfilter(zeroCoeffs, poleCoeffs, signal)
    PlaySound(processedSig, fs)

    PrepareSignalPlot(axesSig[1, 0], processedSig, t, fs, 0.005, 'Processed Signal')
    PrepareFFTPlot(axesSig[1, 1], processedSig, t, fs, 'FFT of Processed Signal')

    plt.tight_layout()

    plt.show()

def PreparePoleZeroPlot(ax, zeroesArray, polesArray, title):
    theta = np.linspace(0, 2 * np.pi, 100)
    unitCircle = np.exp(1j * theta)

    ax.plot(unitCircle.real, unitCircle.imag, 'k--', label='Unit Circle')

    ax.scatter(zeroesArray.real, zeroesArray.imag, marker='o', color='b', label='Zeroes', s=80)
    ax.scatter(polesArray.real, polesArray.imag, marker='x', color='r', label='Poles', s=100)

    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)

    ax.set_title(title)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.legend()
    ax.grid()
    ax.axis('equal')

def PrepareSignalPlot(ax, signalX, signalY, sampleRate, durationToShow, title):
    ax.plot(signalY[:int(sampleRate * durationToShow)], signalX[:int(sampleRate * durationToShow)], color='b')

    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, linestyle='--', alpha=0.6)

def PrepareFFTPlot(ax, signalX, signalY, sampleRate, title):
    result = np.fft.fft(signalX)
    frequencies = np.fft.fftfreq(len(signalY), 1 / sampleRate)
    magDb = (20 * np.log10(np.abs(result)))


    ax.plot(frequencies[:len(frequencies) // 2], magDb[:len(frequencies) // 2])

    # Divide by two because we only care about the positive frequencies
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')

def PrepareFreqRespPlot(ax, w, h, nyquistFreq, title):
    majorGridX = np.arange(0, nyquistFreq + 4000, 4000)
    minorGridX = np.arange(0, nyquistFreq + 2000, 2000)

    db = (20 * np.log10(np.abs(h)))
    maxMag = np.max(db)
    minMag = np.min(db)
    majorGridY = np.linspace(round(minMag / 5) * 5, round(maxMag / 5) * 5, num=10)

    ax.plot(w, db, 'b')

    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_xlim(0, nyquistFreq)
    ax.set_xticks(majorGridX)
    ax.set_yticks(majorGridY)
    ax.grid(which='both', linestyle='--', alpha=0.7)


def PreparePhaseRespPlot(ax, w, h, nyquistFreq, title):
    majorGridX = np.arange(0, nyquistFreq + 4000, 4000)
    minorGridX = np.arange(0, nyquistFreq + 2000, 2000)

    angle = np.angle(h, deg=True)
    maxMag = np.max(angle)
    minMag = np.min(angle)


    majorGridY = np.linspace(round(minMag, -1), round(maxMag, -1), num=10)
    minorGridY = majorGridY // 2

    ax.plot(w, angle, 'g')

    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase (degrees)')
    ax.set_xlim(0, nyquistFreq)
    ax.set_xticks(majorGridX)
    ax.set_yticks(majorGridY)
    ax.grid(which='both', linestyle='--', alpha=0.7)

def PlaySound(signalArray, sampleRate):
    sd.play(signalArray, samplerate=sampleRate)
    sd.wait()

if __name__ == '__main__':
    main()
