##############################################################################
#                                                 
#  David Vorous                                   
#  ECE-351-51                                     
#  Lab 5                                          
#  09-28-21
#  
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14}) # Set font size in plots

steps = 1e-8 # Define step size
t1 = np.arange(0, 1.2e-3 + steps , steps)

# --- User - Defined Functions ---

def reg_h(t):
    return 10355.607 * np.exp(-5000 * t) * np.cos(18584.143 * t + 0.262823)

hs = ([0, 10000, 0],[1, 10000, 370.37037037e6])

impt, impy = sig.impulse(hs, T = t1)
stpt, stpy = sig.step(hs, T = t1)

# --- Plots ---
##############################################################################

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(t1, reg_h(t1))
plt.grid()
plt.ylabel('y = calc h(t)')
plt.title('Plots of Impulse Response to H(s) for t ∈ [0, 0.0012]')

plt.subplot(2, 1, 2)
plt.plot(impt, impy)
plt.grid()
plt.ylabel('y = lib h(t)')
plt.xlabel('t = seconds')

##############################################################################

plt.figure(figsize = (10 , 7))
plt.subplot(1, 1, 1)
plt.plot(stpt, stpy)
plt.grid()
plt.ylabel('y = step_h(t)')
plt.xlabel('t = seconds')
plt.title('Plot of Step Response to H(s) for t ∈ [0, 0.0012]')