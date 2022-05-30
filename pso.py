'''
Author: CLOUDUH
Date: 2022-05-28 17:55:32
LastEditors: CLOUDUH
LastEditTime: 2022-05-29 10:41:14
Description: Battery charging optimization by PSO
    Battery charging optimization program.
    Optimization algorithm is particle swarm optimization
'''

from battery_model import battery_charged

import sys
import numpy as np
import numpy.matlib

t_p = 1 # Step
t_m = 2*3600 # Empirical charging time
w = 0.729 # Inertial weight
c1 = 1.49115 # Self learning factor
c2 = 1.49115 # Swarm learning factor 
beta = 1 # Weight coefficient 1: fastest; 0: healthiest
ger = 50 # The maximum number of iterations 

def particle_swarm_optimization(N:int, d:int, ger:int):
    '''Particle swarm optimization
    Args:
        N: Particle swarm number
        d: Particle dimension
    Returns:
        SoH: SoH of each iteration of each particle (ger-N)
        t: Charging time cost
        ym: The best known position of entire swarm (1-d)
        fym: Global optimal objective function value
        fx: Objective function value of each iteration of each particle (ger-N)
        fxm: Optimal objective function value of particle (N-1)
    '''
    
    iter = 1  # Initial iteration

    x = np.zeros((N,d)) # Pariticle Position (N-d)
    v = np.zeros((N,d)) # Pariticle Velcocity (N-d)

    Ilimit = np.matlib.repmat(np.array([[0],[3.3]]),1,d) # Charging current limits (2-d)
    vlimit = np.matlib.repmat(np.array([[-0.33],[0.33]]),1,d) # Velocity limits (2-d)

    f1 = lambda t: beta * t / t_m # Object function

    for i in range(d):
        if i == 0:
            x[:,i] = np.matlib.repmat(Ilimit[0,i],1,N) + (Ilimit[1,i] - Ilimit[0,i]) * np.random.rand(1,N)
        else:
            x[:,i] = np.matlib.repmat(Ilimit[0,i],1,N) + (x[:,i-1] - Ilimit[0,i]) * np.random.rand(1,N)
        v[:,i] = np.matlib.repmat(vlimit[0,i],1,N) + (vlimit[1,i] - vlimit[0,i]) * np.random.rand(1,N)

    xm = x # The best known position of particle (N-d)
    ym = np.zeros((1,d)) # The best known position of entire swarm (1-d)
    SoH = np.zeros((ger,N)) # SoH of each iteration of each particle (ger-N)
    t = np.zeros((ger,N)) # Charging time of each iteration of each particle (ger-N)
    fx = np.zeros((ger,N)) # Objective function value of each iteration of each particle (ger-N)
    fxm = np.zeros((N,1)) # Optimal objective function value of particle (N-1)
    fym = float("-inf") # Global optimal objective function value

    while iter <= ger:

        for j in range(N):

            [SoH[iter,j],t[iter,j]]= battery_charged(t_p, x[j],iter,j) # Battery simulation
            fx[iter,j] = f1(t[iter,j]) # Optimal function value

            if fxm[j] > fx[iter,j]:
                fxm[j] = fx[iter,j]
                xm[j] = x[j]

        if fym > np.amin(fxm):
            fym = np.amin(fxm)
            nmin = np.argmin(fxm)
            ym = xm(nmin)

        v = v * w + c1 * np.random.rand() * (xm - x) + c2 * np.random.rand() * (np.matlib.repmat(ym,N,1) - x)

        # Position saturation
        for k in range(d):
            hsat = np.where(v[:,k] < vlimit[0,k], 1, 0)
            lsat = np.where(v[:,k] > vlimit[1,k], 2, 0)

            for j in range(N):
                if hsat[j] == 1:
                    v[j,k] = vlimit[0,k]
                elif lsat[j] == 2:
                    v[j,k] = vlimit[1,k]

        x = x + v # Updating position

        # Velcocity saturation
        for k in range(d): 
            hsat = np.where(x[:,k] < Ilimit[0,k], 1, 0)
            lsat = np.where(x[:,k] > Ilimit[1,k], 2, 0)

            for j in range(N):
                if hsat[j] == 1:
                    x[j,k] = Ilimit[0,k]
                elif lsat[j] == 2:
                    x[j,k] = Ilimit[1,k]
    
        iter = iter + 1
        print(iter)

    return [SoH,t,ym,fym,fx,fxm]

if __name__ == '__main__':
    particle_swarm_optimization(20, 5, 50)