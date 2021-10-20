##############################################################################
#                                                 
#  David Vorous                                   
#  ECE-351-51                                     
#  Lab 7                                          
#  10-12-21
#  
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14}) # Set font size in plots

steps = 1e-2 # Define step size
t1 = np.arange(0, 4 + steps, steps)
t2 = np.arange(0, 8 + steps, steps)

# --- User - Defined Functions ---

Gnum = [1,9]
Gden = sig.convolve([1,-8],sig.convolve([1,4],[1,2]))

Anum = [1,4]
Aden = sig.convolve([1,3],[1,1])

Bnum = [1,26,168]

OLnum = [1,9]
OLden = sig.convolve([1,-6,-16],[1,4,3])

CLnum = sig.convolve(Anum, Gnum)
CLden = sig.convolve(Aden, Gden + sig.convolve(Bnum, Gnum))

Gz,Gp,Gk = sig.tf2zpk(Gnum, Gden)
print("G(s) z,p,k = ", Gz, Gp, Gk)

Az,Ap,Ak = sig.tf2zpk(Anum, Aden)
print("\nA(s) z,p,k = ", Az, Ap, Ak)

Bz = np.roots(Bnum)
print("\nB(s) z =     ", Bz)

OLz,OLp,OLk = sig.tf2zpk(OLnum, OLden)
print("\nOL(s) z,p,k = ", OLz, OLp, OLk)
print("Open Loop is unstable because one of its roots is positive")

OLstept, OLstepy = sig.step((OLnum, OLden), T = t1)

CLz,CLp,CLk = sig.tf2zpk(CLnum, CLden)
print("\nCL(s) z,p,k = ", CLz, CLp, CLk)
print("Closed Loop is stable because the real-part of every pole is negative")

CLstept, CLstepy = sig.step((CLnum, CLden), T = t2)

# --- Plots ---

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(OLstept, OLstepy)
plt.grid()
plt.ylabel('y = OL(t)')
plt.title('Plot of Step Response to OL(s) for t ∈ [0, 4]')

plt.figure(figsize = (10 , 7))
plt.subplot(2, 1, 1)
plt.plot(CLstept, CLstepy)
plt.grid()
plt.ylabel('y = CL(t)')
plt.title('Plot of Step Response to CL(s) for t ∈ [0, 8]')

