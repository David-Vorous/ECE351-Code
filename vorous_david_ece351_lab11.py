##############################################################################
#                                                 
#  David Vorous                                   
#  ECE-351-51                                     
#  Lab 11                                         
#  11-16-21
#  
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14}) # Set font size in plots

# --- Basic Variable Setup ---
Hnum = [2,-40, 0]
Hden = [1,-10,16]

r1 = [6, -4]
p1 = [2,  8]


# --- User - Defined Functions ---
def dB(x):
    return 20*np.log10(x)

# --- Provided Functions ---
def zplane(b, a, zeros, filename):
    from matplotlib import patches
    
    # get a figure/plot
    ax = plt.subplot(1,1,1)
    
    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False, color='black', ls='dashed')
    ax.add_patch(uc)
    
    # the coefficients are less than 1, normalize the coefficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = np.array(b)/float(kn)
    else:
        kn = 1
    if np.max(a) > 1:
        kd = np.max(a)
        a = np.array(a)/float(kd)
    else:
        kd = 1
    
    # get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # plot the zeros and set marker properties
    if zeros:
        t1 = plt.plot(z.real, z.imag, 'o', ms=10, label='Zeros')
        plt.setp(t1, markersize=10.0, markeredgewidth=1.0)
    
    # plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'x', ms=10, label='Poles')
    plt.setp(t2, markersize=12.0, markeredgewidth=3.0)
    
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.legend()
    
    # set the ticks   
    # r = 20; plt.axis('scaled'); plt.axis([-r,r,-r,r])
    # ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        
    return z,p,k


# --- Prints ---
r2,p2,k2 = sig.residuez(Hnum, Hden)
print ("hand-calculated residues:", r1,"\nsig.residuez() residues: ", r2, "\n")
print ("hand-calculated poles:   ", p1,"\nsig.residuez() poles:    ", p2, "\n")

r3,p3,k3 = zplane(Hnum, Hden, True, None)
zplane(Hnum, Hden, False, None)
print ("zplane() roots:          ", r3,"\nzplane() poles:          ", p3, "\n")

w,h = sig.freqz(Hnum, Hden, worN=512, whole=True, plot=None, fs=2*np.pi, include_nyquist=False)

magdB = dB(np.abs(h))
phase = np.angle(h) * 180.0/np.pi


# --- Plots ---
plt.figure(figsize = (8 , 6))
plt.subplot(2, 1, 1)
plt.plot(w * 0.5/np.pi, magdB)
plt.grid()
plt.ylabel('|H(z)| dB')
plt.title('Plots of H(z) for z ∈ [0, 1] Hz')

plt.subplot(2, 1, 2)
plt.plot(w * 0.5/np.pi, phase)
plt.grid()
plt.ylabel('/_H(z)°')
plt.xlabel('z (Hz)')
