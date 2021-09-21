###################################################
#                                                 
#  David Vorous                                   
#  ECE-351-51                                     
#  Lab 3                                          
#  09-14-21
#  
###################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14}) # Set font size in plots

steps = 1e-2 # Define step size
t1 = np.arange(0, 20 + steps , steps)

print ('\nNumber of elements : len(t) = ', len(t1), '\nFirst Element : t[0] = ',
       t1[0] , '\nLast Element : t1[len(t) - 1] = ', t1[len(t1) - 1])
# Notice the array might be a different size than expected since Python starts
# at 0. Then we will use our knowledge of indexing to have Python print the
# first and last index of the array . Notice the array goes from 0 to len () - 1

# --- User - Defined Function ---

def step(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

def ramp(t):
    y = np.zeros(t.shape) 
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = t[i]
        else:
            y[i] = 0
    return y

def conv(f1,f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1Extended = np.append(f1, np.zeros((1, Nf2 - 1)))
    f2Extended = np.append(f2, np.zeros((1, Nf1 - 1)))
    result = np.zeros(f1Extended.shape)
    for i in range(Nf2 + Nf1 - 2):
        result[i] = 0
        for j in range(Nf1):
            if ((i - j + 1) > 0):
                try:
                    result[i] += f1Extended[j] * f2Extended[i - j + 1]
                except:
                    print(i,j)
    return (result * steps)

def func1(t):
    return step(t-2) - step(t-9)

def func2(t):
    return np.exp(-t) * step(t)

def func3(t):
    return ramp(t-2) * (step(t-2) - step(t-3)) + ramp(4-t) * (step(t-3) - step(t-4))

t2 = np.arange(0, 40 + 2 * steps , steps)

################################################################
plt.figure(figsize = (10 , 7))
plt.subplot(3, 1, 1)
plt.plot(t1, func1(t1))
plt.grid()
plt.ylabel('y(t) = f1(t)')
plt.title('Plot of f1(t), f2(t), f3(t) for t ∈ [0,20]')

plt.subplot(3, 1, 2)
plt.plot(t1, func2(t1))
plt.grid()
plt.ylabel('y(t) = f2(t)')

plt.subplot(3, 1, 3)
plt.plot(t1, func3(t1))
plt.grid()
plt.ylabel('y(t) = f3(t)')
plt.xlabel('t = Seconds')

################################################################
plt.figure(figsize = (10 , 7))
plt.subplot(3, 1, 1)
plt.plot(t2, conv(func1(t1),func2(t1)))
plt.grid()
plt.ylabel('Hand-Coded Convolve', fontsize=11)
plt.title('Plot of f1(t)*f2(t) for t ∈ [0,40]')

plt.subplot(3, 1, 2)
plt.plot(t2, steps * sig.convolve(func1(t1),func2(t1)))
plt.grid()
plt.ylabel('Library Convolve', fontsize=11)

################################################################
plt.figure(figsize = (10 , 7))
plt.subplot(3, 1, 1)
plt.plot(t2, conv(func2(t1),func3(t1)))
plt.grid()
plt.ylabel('Hand-Coded Convolve', fontsize=11)
plt.title('Plot of f2(t)*f3(t) for t ∈ [0,40]')

plt.subplot(3, 1, 2)
plt.plot(t2, steps * sig.convolve(func2(t1),func3(t1)))
plt.grid()
plt.ylabel('Library Convolve', fontsize=11)

################################################################
plt.figure(figsize = (10 , 7))
plt.subplot(3, 1, 1)
plt.plot(t2, conv(func3(t1),func1(t1)))
plt.grid()
plt.ylabel('Hand-Coded Convolve', fontsize=11)
plt.title('Plot of f3(t)*f1(t) for t ∈ [0,40]')

plt.subplot(3, 1, 2)
plt.plot(t2, steps * sig.convolve(func3(t1),func1(t1)))
plt.grid()
plt.ylabel('Library Convolve', fontsize=11)