##############################################################################
#                                                 
#  David Vorous                                   
#  ECE-351-51                                     
#  Lab 6                                          
#  10-05-21
#  
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14}) # Set font size in plots

steps = 1e-2 # Define step size
t1 = np.arange(0, 2 + steps , steps)
t2 = np.arange(0, 4.5 + steps , steps)

# --- User - Defined Functions ---

def u(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

def invLap(num, den, t):
    r,p,k = sig.residue(num, den)
    print(r,p,k)
    
    Combine = 0
    for i in range(len(r)):
        magn = np.abs(r[i])
        angl = np.angle(r[i])
        alph = np.real(p[i])
        omeg = np.imag(p[i])
        Combine += (magn * np.exp(alph * t) * np.cos((omeg * t) + angl)) * u(t) 
    return Combine

def reg_y(t):
    return (0.5 - 0.5 * np.exp(-4.0 * t) + np.exp(-6.0 * t)) * u(t)

hs = ([1, 6, 12],[1, 10, 24])

stept, stepy = sig.step(hs, T = t1)

ys1_num = [0, 1,  6, 12]
ys1_den = [1, 10, 24, 0]

r1,p1,k1 = sig.residue(ys1_num, ys1_den)
print(r1,p1,k1)

ys2_num = [0,0, 0,  0,   0,   0,25250]
ys2_den = [1,18,218,2036,9085,25250,0]

yt2 = invLap(ys2_num, ys2_den, t2)

ys3_num = [0,0, 0,  0,   0,   25250]
ys3_den = [1,18,218,2036,9085,25250]

libt, liby = sig.step((ys3_num,ys3_den), T = t2)

# --- Plots ---
##############################################################################

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(t1, reg_y(t1))
plt.grid()
plt.ylabel('y = calc y(t)')
plt.title('Plots of Step Response to H(s) for t ∈ [0, 2]')

plt.subplot(2, 1, 2)
plt.plot(stept, stepy)
plt.grid()
plt.ylabel('y = lib y(t)')
plt.xlabel('t = seconds')

##############################################################################

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(t2, yt2)
plt.grid()
plt.ylabel('y = calc y(t)')
plt.title('Plots of Step Response to H(s) for t ∈ [0, 2]')

plt.subplot(2, 1, 2)
plt.plot(libt, liby)
plt.grid()
plt.ylabel('y = lib y(t)')
plt.xlabel('t = seconds')
