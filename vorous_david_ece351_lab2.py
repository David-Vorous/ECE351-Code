###################################################
#                                                 
#  David Vorous                                   
#  ECE-351-51                                     
#  Lab 2                                          
#  09-07-21
#  
###################################################

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14}) # Set font size in plots

steps = 1e-2 # Define step size
t1 = np.arange(0, 10 + steps , steps)
t2 = np.arange(-5, 10 + steps, steps)

print ('\nNumber of elements : len(t) = ', len(t1), '\nFirst Element : t[0] = ',
       t1[0] , '\nLast Element : t1[len(t) - 1] = ', t1[len(t1) - 1])
# Notice the array might be a different size than expected since Python starts
# at 0. Then we will use our knowledge of indexing to have Python print the
# first and last index of the array . Notice the array goes from 0 to len () - 1

# --- User - Defined Function ---

# Create output y(t) using a for loop and if/ else statements
def func1(t): # The only variable sent to the function is t
    y = np.zeros(t.shape) # initialze y(t) as an array of zeros

    for i in range(len(t)): # run the loop once for each index of t
        y[i] = np.cos(t[i])
    return y # send back the output stored in an array

def func2(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

def func3(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = t[i]
        else:
            y[i] = 0
    return y

def func4(t):
    y = func3(t) - func3(t-3) + 5*func2(t-3) - 2*func2(t-6) - 2*func3(t-6)
    return y

def func5(f, t):
    y = np.diff(f) / np.diff(t)       # ensure that f is a function of t
    return y


cos = func1(t1) # call the function we just created
step = func2(t1 - 1)
ramp = func3(t1 - 2)
piece1 = func4(t2)
deriv = func5(func4(t2), t2)


plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(t1, cos)
plt.grid()
plt.ylabel('y(t) = cos(t)')
plt.xlabel('t = Seconds')
plt.title('Plot of cos(t) for t ∈ [0,10]')

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(t1, step)
plt.grid()
plt.ylabel('y(t) = step(t - 1)')
plt.xlabel('t = Seconds')
plt.title('Plot of step(t - 1) for t ∈ [0,10]')

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(t1, ramp)
plt.grid()
plt.ylabel('y(t) = ramp(t - 2)')
plt.xlabel('t = Seconds')
plt.title('Plot of ramp(t - 2) for t ∈ [0,10]')

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(t2, piece1)
plt.grid()
plt.ylabel('y(t) = piece(t)')
plt.xlabel('t = Seconds')
plt.title('Plot of piece(t) for t ∈ [-5,10]')

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(-t2, piece1)
plt.grid()
plt.ylabel('y(t) = piece(-t)')
plt.xlabel('t = Seconds')
plt.title('Plot of piece(-t) for t ∈ [-10,5]')

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(t2 - 4, piece1)
plt.grid()
plt.ylabel('y(t) = piece(t - 4)')
plt.xlabel('t = Seconds')
plt.title('Plot of piece(t - 4) for t ∈ [-10,6]')

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(-t2 - 4, piece1)
plt.grid()
plt.ylabel('y(t) = piece(-t - 4)')
plt.xlabel('t = Seconds')
plt.title('Plot of piece(-t - 4) for t ∈ [-14,1]')

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(t2/2, piece1)
plt.grid()
plt.ylabel('y(t) = piece(t/2)')
plt.xlabel('t = Seconds')
plt.title('Plot of piece(t/2) for t ∈ [-3,5]')

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(t2 * 2, piece1)
plt.grid()
plt.ylabel('y(t) = piece(t * 2)')
plt.xlabel('t = Seconds')
plt.title('Plot of piece(t * 2) for t ∈ [-10,20]')

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(t2[range(len(deriv))], deriv)
plt.grid()
plt.ylim([-2, 4])
plt.ylabel("y'(t) = d(piece(t))/dt")
plt.xlabel('t = Seconds')
plt.title('Plot of d(piece(t))/dt for t ∈ [-5,10]')