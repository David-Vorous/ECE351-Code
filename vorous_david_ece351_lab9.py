##############################################################################
#                                                 
#  David Vorous                                   
#  ECE-351-51                                     
#  Lab 9                                         
#  10-26-21
#  
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

plt.rcParams.update({'font.size': 14}) # Set font size in plots

steps = 1e-2 # Define step size
t1 = np.arange(0, 2, steps)
t2 = np.arange(0, 16, steps)

# --- User - Defined Functions ---

x1 = np.cos(2 * np.pi * t1)
x2 = 5 * np.sin(2 * np.pi * t1)
x3 = 2 * np.cos((2*np.pi * 2*t1) - 2) + (np.sin((2*np.pi * 6*t1) + 3))**2

def w(T):
    return ((2 * np.pi)/T)
def bk(k):
    return ((2.0/(k * np.pi)) * (1 - np.cos(k * np.pi)))

def x(t, T, N):
    y = 0
    for k in np.arange(1, N + 1):
        y += ((np.sin(k * w(T) * t)) * bk(k))
    return y

x4 = x(t2, 8, 15)

def FFT1(x,fs):
    N = len(x)                        # find the length of the signal
    X_fft = scipy.fftpack.fft(x)      # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift( X_fft ) # shift zero frequency components
                                                    # to the center of the spectrum             
    freq = np.arange(-N/2, N/2)*fs/N  # compute the frequencies for the output
                                      # signal , (fs is the sampling frequency and
                                      # needs to be defined previously in your code
    X_mag = np.abs( X_fft_shifted )/N # compute the magnitudes of the signal

    X_phi = np.angle( X_fft_shifted ) # compute the phases of the signal
    
    return freq, X_mag, X_phi

def FFT2(x,fs):
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

def FFT_plot1(t, freq, x, X_mag, X_phi):
    plt.figure(figsize = (10,7))
    plt.subplot(3,1,1)
    plt.plot(t,x)
    plt.grid()
    plt.xlabel('t(s)')
    plt.ylabel('x(t)')
    plt.title('FFT')
    
    plt.subplot(3,2,3)
    plt.stem(freq, X_mag)
    plt.grid()
    plt.ylabel('|X(f)|')
    
    plt.subplot(3,2,4)
    plt.stem(freq, X_mag)
    plt.grid()
    plt.xlim([-2,2])
    
    plt.subplot(3,2,5)
    plt.step(freq, X_phi)
    plt.grid()
    plt.ylabel('Phase X(f)')
    plt.xlabel('f(Hz)')
    
    plt.subplot(3,2,6)
    plt.stem(freq, X_phi)
    plt.grid()
    plt.xlim([-2,2])
    plt.xlabel('f(Hz)')
    
    plt.tight_layout()
    plt.show()
    
def FFT_plot2(t, freq, x, X_mag, X_phi):
    plt.figure(figsize = (10,7))
    plt.subplot(3,1,1)
    plt.plot(t,x)
    plt.grid()
    plt.xlabel('t(s)')
    plt.ylabel('x(t)')
    plt.title('FFT')
    
    plt.subplot(3,2,3)
    plt.stem(freq, X_mag)
    plt.grid()
    plt.ylabel('|X(f)|')
    
    plt.subplot(3,2,4)
    plt.stem(freq, X_mag)
    plt.grid()
    plt.xlim([-15,15])
    
    plt.subplot(3,2,5)
    plt.step(freq, X_phi)
    plt.grid()
    plt.ylabel('Phase X(f)')
    plt.xlabel('f(Hz)')
    
    plt.subplot(3,2,6)
    plt.stem(freq, X_phi)
    plt.grid()
    plt.xlim([-15,15])
    plt.xlabel('f(Hz)')
    
    plt.tight_layout()
    plt.show()


# -- Without De-Noiser:
    
freq1, X_mag1, X_phi1 = FFT1(x1, 1/steps)
FFT_plot1(t1, freq1, x1, X_mag1, X_phi1)

freq2, X_mag2, X_phi2 = FFT1(x2, 1/steps)
FFT_plot1(t1, freq2, x2, X_mag2, X_phi2)

freq3, X_mag3, X_phi3 = FFT1(x3, 1/steps)
FFT_plot2(t1, freq3, x3, X_mag3, X_phi3)

freq4, X_mag4, X_phi4 = FFT1(x4, 1/steps)
FFT_plot1(t2, freq4, x4, X_mag4, X_phi4)

# -- With De-Noiser:

freq1, X_mag1, X_phi1 = FFT2(x1, 1/steps)
FFT_plot1(t1, freq1, x1, X_mag1, X_phi1)

freq2, X_mag2, X_phi2 = FFT2(x2, 1/steps)
FFT_plot1(t1, freq2, x2, X_mag2, X_phi2)

freq3, X_mag3, X_phi3 = FFT2(x3, 1/steps)
FFT_plot2(t1, freq3, x3, X_mag3, X_phi3)

freq4, X_mag4, X_phi4 = FFT2(x4, 1/steps)
FFT_plot1(t2, freq4, x4, X_mag4, X_phi4)

