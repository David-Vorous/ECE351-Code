##############################################################################
#                                                 
#  David Vorous                                   
#  ECE-351-51                                     
#  Lab Project                                         
#  11-16-21
#  
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack
import control as con
import pandas as pd

plt.rcParams.update({'font.size': 14}) # Set font size in plots

# --- Basic Variables/Arrays ---
fs = 1e6

df = pd.read_csv('NoisySignal.csv')
t = df['0'].values
sensor_sig = df['1'].values

B = 728 * 2 * np.pi
w0 = 1900 * 2 * np.pi

R = 20.0
L = R/B
C = B/(w0**2 * R)

print (R, L, C)

Hnum = [0, R/L, 0]
Hden = [1, R/L, 1/(L*C)]

# --- User-Defined Functions ---
def FFT(x,fs):
    N = len(x)                        # find the length of the signal
    X_fft = scipy.fftpack.fft(x)      # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift( X_fft ) # shift zero frequency components
                                                    # to the center of the spectrum             
    freq = np.arange(-N/2, N/2)*fs/N  # compute the frequencies for the output
                                      # signal , (fs is the sampling frequency and
                                      # needs to be defined previously in your code
    X_mag = np.abs( X_fft_shifted )/N # compute the magnitudes of the signal

    X_phi = np.angle( X_fft_shifted ) # compute the phases of the signal
    
    for i in range(len(X_phi)):
        if (X_mag[i] < 1e-10):
            X_phi[i] = 0
    
    return freq, X_mag, X_phi

def Hmag(w):
    return np.abs(B*w)/np.sqrt((B*w)**2 + (-w**2 + w0**2)**2)

def PLT_FFT(l, h):
    plt.figure(figsize = (10, 7))
    plt.subplot(2,1,1)
    plt.semilogx(freq, inMAG)
    plt.semilogx(freq * 1/(2*np.pi), Hmag(freq))
    plt.xlim((l,h))
    plt.grid()
    plt.ylabel('prefiltered [mag]')
    plt.title('|X(f)| and |H(f)| for f ∈ [{}, {}]'.format(l, h))
    plt.subplot(2,1,2)
    plt.semilogx(freq, outMAG)
    plt.semilogx(freq * 1/(2*np.pi), Hmag(freq))
    plt.xlim((l,h))
    plt.grid()
    plt.ylabel('filtered [mag]')
    plt.xlabel('f [Hz]')

# --- Calculations ---
freq, inMAG, inPHI = FFT(sensor_sig, fs)

znum, zden = sig.bilinear(Hnum, Hden, fs)
filt = sig.lfilter(znum, zden, sensor_sig)

_, outMAG, outPHI = FFT(filt, fs)

# --- Plots ---
plt.figure(figsize = (10, 7))
plt.plot(t, sensor_sig)
plt.grid()
plt.title('Noisy Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()

plt.figure(figsize = (10, 7))
plt.plot(t, filt)
plt.grid()
plt.title('Filtered Output Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()

sys = con.TransferFunction(Hnum, Hden)
plt.figure(figsize = (10, 7))
con.bode(sys, freq, dB = True, Hz = True, deg = True, Plot = True)

plt.figure(figsize = (10, 7))
con.bode(sys, np.arange(1, 1800, 50) * 2*np.pi, dB = True, Hz = True, deg = True, Plot = True)

plt.figure(figsize = (10, 7))
con.bode(sys, np.arange(2000, 100000, 50) * 2*np.pi, dB = True, Hz = True, deg = True, Plot = True)

plt.figure(figsize = (10, 7))
con.bode(sys, np.arange(1800, 2010, 10) * 2*np.pi, dB = True, Hz = True, deg = True, Plot = True)

PLT_FFT(10, 100000)
PLT_FFT(1, 1800)
PLT_FFT(2000, 100000)
PLT_FFT(1800, 2000)