##############################################################################
#                                                 
#  David Vorous                                   
#  ECE-351-51                                     
#  Lab 10                                          
#  11-02-21
#  
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con

plt.rcParams.update({'font.size': 14}) # Set font size in plots

steps1 = 1
steps2 = 3.183e-6
w = np.arange(1e3, 1e6 + steps1, steps1)
t = np.arange(0, 1e-2 + steps2, steps2)

r = 1e3
c = 100e-9
l = 27e-3

# --- User - Defined Functions ---
def mag_H(w, r, c, l):
    return (w/(r*c))/np.sqrt(((1/(l*c)) - w**2)**2 + (w/(r*c))**2)

def dB(x):
    return 20*np.log10(x)

def ang_H(w, r, c, l):
    return 90 - np.arctan2((w/(r*c)),((1/(l*c)) - w**2))*(180/np.pi)

def x(t):
    return np.cos(200*np.pi*t) + np.cos(6048*np.pi*t) + np.sin(10e5*np.pi*t)

Hnum = [0, 1/(r*c), 0]
Hden = [1, 1/(r*c), 1/(l*c)]
sys = con.TransferFunction(Hnum, Hden)
_ = con.bode(sys, w, dB = True, Hz = True, deg = True, Plot = True)

znum, zden = sig.bilinear(Hnum, Hden, 1/steps2)
filt = sig.lfilter(znum, zden, x(t))

# --- Plots ---
plt.figure(figsize = (8 , 6))
plt.subplot(2, 1, 1)
plt.semilogx(w, dB(mag_H(w, r, c, l)))
plt.grid()
plt.ylabel('|H(w)| dB')
plt.title('Plots of H(w) for w ∈ [10^3, 10^6]')

plt.subplot(2, 1, 2)
plt.semilogx(w, ang_H(w, r, c, l))
plt.grid()
plt.ylabel('/_H(w)°')
plt.xlabel('w (rad/s)')

plt.figure(figsize = (8 , 6))
plt.subplot(2, 1, 1)
plt.plot(t, x(t))
plt.grid()
plt.ylabel('x(t)')
plt.title('Plot of x(t) for t ∈ [0, 0.01]')

plt.figure(figsize = (8 , 6))
plt.subplot(2, 1, 1)
plt.plot(t, filt)
plt.grid()
plt.ylabel('Fx(t)')
plt.title('Plot of Filtered x(t) for t ∈ [0, 0.01]')

