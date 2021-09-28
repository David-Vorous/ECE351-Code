##############################################################################
#                                                 
#  David Vorous                                   
#  ECE-351-51                                     
#  Lab 4                                          
#  09-21-21
#  
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14}) # Set font size in plots

steps = 1e-2 # Define step size
t1 = np.arange(-10, 10 + steps , steps)

# --- Basic Functions ---

def u(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

def r(t):
    y = np.zeros(t.shape) 
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = t[i]
        else:
            y[i] = 0
    return y

# --- User - Defined Functions ---

def h1(t):
    return np.exp(-2 * t) * (u(t) - u(t - 3))

def h2(t):
    return u(t-2) - u(t-6)

def h3(t):
    return np.cos(4 * t) * u(t)

def step_h1(t):
    return 0.5*(1-np.exp(-2*t))*u(t) - 0.5*(1-np.exp(-2*(t-3)))*u(t-3)

def step_h2(t):
    return r(t-2) - r(t-6)

def step_h3(t):
    return 0.25 * np.sin(4*t)*u(t)

print(len(t1), t1[2000])

# --- Plots ---
##############################################################################

plt.figure(figsize = (10 , 7))
plt.subplot(3, 1, 1)
plt.plot(t1, h1(t1))
plt.grid()
plt.ylabel('y = h1(t)')
plt.title('Plot of h1(t), h2(t), h3(t) for t ∈ [-10,10]')

plt.subplot(3, 1, 2)
plt.plot(t1, h2(t1))
plt.grid()
plt.ylabel('y = h2(t)')

plt.subplot(3, 1, 3)
plt.plot(t1, h3(t1))
plt.grid()
plt.ylabel('y = h3(t)')
plt.xlabel('t = Seconds')

##############################################################################

t2 = np.arange(-20, 19.9999995 + steps, steps)

##############################################################################

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(t2, steps * sig.convolve(h1(t1),u(t1)))
plt.grid()
plt.xlim(-10,20)
plt.ylim(-0.1,0.6)
plt.ylabel('Library Conv', fontsize=11)
plt.title('Plot of  (h1*u)(t) for t ∈ [-10,10] or t ∈ [-10,20]')

plt.subplot(2, 1, 2)
plt.plot(t1, step_h1(t1))
plt.grid()
plt.xlim(-10,10)
plt.ylim(-0.1,0.6)
plt.ylabel('Integrated', fontsize=11)

##############################################################################

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(t2, steps * sig.convolve(h2(t1),u(t1)))
plt.grid()
plt.xlim(-10,20)

plt.ylabel('Library Conv', fontsize=11)
plt.title('Plot of  (h2*u)(t) for t ∈ [-10,10] or t ∈ [-10,20]')

plt.subplot(2, 1, 2)
plt.plot(t1, step_h2(t1))
plt.grid()
plt.xlim(-10,10)

plt.ylabel('Integrated', fontsize=11)

##############################################################################

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(t2, steps * sig.convolve(h3(t1),u(t1)))
plt.grid()
plt.xlim(-10,20)

plt.ylabel('Library Conv', fontsize=11)
plt.title('Plot of  (h3*u)(t) for t ∈ [-10,10] or t ∈ [-10,20]')

plt.subplot(2, 1, 2)
plt.plot(t1, step_h3(t1))
plt.grid()
plt.xlim(-10,10)

plt.ylabel('Integrated', fontsize=11)

##############################################################################
