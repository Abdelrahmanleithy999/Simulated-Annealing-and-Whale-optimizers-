## Generate a contour plot
# Import some other libraries that we'll need
# matplotlib and numpy packages must also be installed
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# define objective function
def f(x):
    zd = x[0]
    zb = x[1]
    za = x[2]
    zf = x[3]

    itot = (zd * zb) / (za * zf)
    itarget = 1 / 6.931

    obj = (itarget - itot) * (itarget - itot)
    return obj

# Start location
x_start = [12, 12, 12, 12]

 

##################################################
# Simulated Annealing
##################################################
# Number of cycles
n = 1000
# Number of trials per cycle
m = 1
# Number of accepted solutions
na = 0.0
# Probability of accepting worse solution at the start
p1 = 0.7
# Probability of accepting worse solution at the end
#p50 = 0.001
#upper and lower boundries 
lower = 12
upper = 60


# Initial temperature
t_init = 10000            #-1.0/math.log(p1)
# Final temperature
t_final = 20             #-1.0/math.log(p50)

#linear temperature coeffecient 
beta = 0.02
#alpha = 0.999


# Initialize x
x = np.zeros((n+1,4))
x[0] = x_start



xi = np.zeros(4)
xi = x_start
na = na + 1.0


# Current best results so far
xc = np.zeros(4)
xc = x[0]
fc = f(xi)
fs = np.zeros(n+1)
fs[0] = fc

# Current temperature
t = t_init
# DeltaE Average
DeltaE_avg = 0.0




for i in range(n):
    print('Cycle: ' + str(i) + ' with Temperature: ' + str(t))
    for j in range(m):
        # Generate new trial points
        xi[0] = xc[0] + random.randint(-1,1)
        xi[1] = xc[1] + random.randint(-1,1)
        xi[2] = xc[2] + random.randint(-1,1)
        xi[3] = xc[3] + random.randint(-1,1)

        #rounding the variables up to integer 
        xi[0] = math.ceil(xi[0])
        xi[1] = math.ceil(xi[1])
        xi[2] = math.ceil(xi[2])
        xi[3] = math.ceil(xi[3]) 

        # Clip to upper and lower bounds
        xi[0] = max(min(xi[0],upper),lower)
        xi[1] = max(min(xi[1],upper),lower)
        xi[2] = max(min(xi[2],upper),lower)
        xi[3] = max(min(xi[3],upper),lower)

        DeltaE = abs(f(xi)-fc)

        if (f(xi)>fc):

            p = math.exp(-DeltaE/t)
            # determine whether to accept worse point
            if (random.random()<p):
                # accept the worse solution
                accept = True
            else:
                # don't accept the worse solution
                accept = False
        else:
            # objective function is lower, automatically accept
            accept = True


        if (accept==True):
            # update currently accepted solution
            xc[0] = xi[0]
            xc[1] = xi[1]
            xc[2] = xi[2]
            xc[3] = xi[3]
            fc = f(xc)
            # increment number of accepted solutions
            na = na + 1.0
    # Record the best x values at the end of every cycle
    x[i+1][0] = xc[0]
    x[i+1][1] = xc[1]
    x[i+1][2] = xc[2]
    x[i+1][3] = xc[3]
    fs[i+1] = fc
    # Lower the temperature for next cycle
    t =  t - beta * i
    #t = t * alpha**i 

    if (t <= t_final):
        break


# print solution
print('Best solution: ' + str(xc))
print('Best objective: ' + str(fc))

#plt.plot(x[:,0],x[:,1],'y-o')
#plt.savefig('contour.png')

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(fs[0:i+1],'r.-')
ax1.legend(['Objective'])

#ax2 = fig.add_subplot(512)
#ax2.plot(x[0:i,0],'b.-')
#ax2.legend(['zd'])

#ax3 = fig.add_subplot(513)
#ax3.plot(x[0:i,1],'g.-')
#ax3.legend(['zb'])

#ax4 = fig.add_subplot(514)
#ax4.plot(x[0:i,2],'y.-')
#ax4.legend(['za'])

#ax5 = fig.add_subplot(515)
#ax5.plot(x[0:i,3],'r.-')
#ax5.legend(['zf'])




#ax2.plot(x[0:i,1],'g--')
#ax2.plot(x[0:i,2],'y.-')
#ax2.plot(x[0:i,3],'r.-')
#ax2.legend(['zd','zb','za','zf'])

# Save the figure as a PNG
plt.savefig('iterations.png')

plt.show()
