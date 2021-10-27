##############################################################################
#                                                 
#  David Vorous                                   
#  ECE-351-51                                     
#  Lab 8                                          
#  10-19-21
#  
##############################################################################

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14}) # Set font size in plots

steps = 1e-5 # Define step size
t1 = np.arange(0, 20 + steps, steps)


# --- User - Defined Functions ---
def w(T):
    return ((2 * np.pi)/T)
def bk(k):
    return ((-2.0/(k * np.pi)) * (np.cos(k * np.pi) - 1))

print (bk(1), bk(2), bk(3))

def x(t, T, N):
    y = 0
    for k in np.arange(1, N + 1):
        y += ((np.sin(k * w(T) * t)) * bk(k))
    return y


# --- Plots ---
plt.figure(figsize = (10 , 16))
plt.subplot(6, 1, 1)
plt.plot(t1, x(t1, 8, 1))
plt.grid()
plt.ylabel('y = x(t, N = 1)')
plt.title('Plots of x(t) for t ∈ [0, 20]')

plt.subplot(6, 1, 2)
plt.plot(t1, x(t1, 8, 3))
plt.grid()
plt.ylabel('y = x(t, N = 3)')

plt.subplot(6, 1, 3)
plt.plot(t1, x(t1, 8, 15))
plt.grid()
plt.ylabel('y = x(t, N = 15)')

plt.subplot(6, 1, 4)
plt.plot(t1, x(t1, 8, 50))
plt.grid()
plt.ylabel('y = x(t, N = 50)')

plt.subplot(6, 1, 5)
plt.plot(t1, x(t1, 8, 150))
plt.grid()
plt.ylabel('y = x(t, N = 150)')

plt.subplot(6, 1, 6)
plt.plot(t1, x(t1, 8, 1500))
plt.grid()
plt.ylabel('y = x(t, N = 1500)')
#--------------------------------------------------------------
plt.figure(figsize = (10 , 4))
plt.subplot(1, 1, 1)
plt.plot(t1, x(t1, 8, 1))
plt.grid()
plt.ylabel('y = x(t, N = 1, 3, 15, 50, 150, 1500)')
plt.title('Plots of x(t) for t ∈ [0, 20]')

plt.subplot(1, 1, 1)
plt.plot(t1, x(t1, 8, 3))

plt.subplot(1, 1, 1)
plt.plot(t1, x(t1, 8, 15))

plt.subplot(1, 1, 1)
plt.plot(t1, x(t1, 8, 50))

plt.subplot(1, 1, 1)
plt.plot(t1, x(t1, 8, 150))

plt.subplot(1, 1, 1)
plt.plot(t1, x(t1, 8, 1500))
