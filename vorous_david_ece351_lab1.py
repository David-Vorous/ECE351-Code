###################################################
#                                                 
#  David Vorous                                   
#  ECE-351-51                                     
#  Lab 2                                          
#  09-07-21
#  
###################################################

import numpy
import matplotlib.pyplot

plt. rcParams . update ({ 'fontsize ': 14}) # Set font size in plots

steps = 1e -2 # Define step size
t = np. arange (0, 5 + steps , steps ) # Add a step size to make sure the
                                       # plot includes 5.0. Since np. arange () only
                                       # goes up to , but doesn 't include the
                                       # value of the second argument
print ('Number of elements : len(t) = ', len(t), '\ nFirst Element : t[0] = ', t[0] , '
    \ nLast Element : t[len(t) - 1] = ', t[len(t) - 1])
# Notice the array might be a different size than expected since Python starts
# at 0. Then we will use our knowledge of indexing to have Python print the
# first and last index of the array . Notice the array goes from 0 to len () - 1

# --- User - Defined Function ---

# Create output y(t) using a for loop and if/ else statements
def example1 (t): # The only variable sent to the function is t
    y = np. zeros (t. shape ) # initialze y(t) as an array of zeros

    for i in range (len (t)): # run the loop once for each index of t
        if i < ( len(t) + 1) /3:
    y[i] = t[i ]**2
else :
y[i] = np.sin (5*t[i]) + 2
return y # send back the output stored in an array

y = example1 (t) # call the function we just created


plt. figure ( figsize = (10 , 7))
plt. subplot (2, 1, 1)
plt. plot (t, y)
plt. grid ()
plt. ylabel ('y(t) with Good Resolution ')
plt. title ('Background - Illustration of for Loops and if/ else Statements ')

t = np. arange (0, 5 + 0.25 , 0.25) # redefine t with poor resolution
y = example1 (t)

plt. subplot (2, 1, 2)
plt. plot (t, y)
plt. grid ()
plt. ylabel ('y(t) with Poor Resolution ')
plt. xlabel ('t')
plt. show ()