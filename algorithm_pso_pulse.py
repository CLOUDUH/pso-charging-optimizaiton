'''
Author: CLOUDUH
Date: 2022-05-28 17:55:32
LastEditors: CLOUDUH
LastEditTime: 2022-06-22 22:48:47
Description: Battery charging optimization by PSO
    Battery charging optimization program.
    Optimization algorithm is particle swarm optimization
'''

from cmath import inf
from model_pulse import battery_pulse_charged
from model_photovoltaic import photovoltaic_model
from model_photovoltaic import irradiation_cal
from model_mppt import mppt_cal
from model_load import load_model

import sys
import numpy as np
import numpy.matlib

def object_func(SoH:float, t:float, policy:list, beta:float):

    t_m = 2*3600 # Empirical charging time

    J = beta * t / t_m + (1 - beta) * SoH

    if t < 2 * 3600 or t > 10 * 3600: J = inf

    return J

def remain_power_calc(date:int, latitude:float):

    t = 0
    Temp = 298.15
    cur_bat = 3.0

    while t <= 24:

        rad = irradiation_cal(t, date, latitude)
        [cur, pwr_k1, pwr_mp] = photovoltaic_model(rad, Temp, volt_k1)
        [volt_k1, volt_k0, cur_bat] = mppt_cal(volt_k0, volt_k1, pwr_k0, pwr_k1, volt_bat)

    return 


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

    w = 0.729 # Inertial weight
    c1 = 1.49115 # Self learning factor
    c2 = 1.49115 # Swarm learning factor 
    beta = 1 # Weight coefficient 1: fastest; 0: healthiest
    ger = 50 # The maximum number of iterations 
    iter = 1  # Initial iteration

    x = np.zeros((N,d)) # Particle Position (N-d)
    v = np.zeros((N,d)) # Particle Velcocity (N-d)

    xlimit_cc = np.matlib.repmat(np.array([[0],[3.3]]),1,d-1) 
    xlimit_pulse = np.array([[0],[6.6]])
    xlimit = np.hstack((xlimit_cc, xlimit_pulse)) # Charging current limits (2-d)

    vlimit_cc = np.matlib.repmat(np.array([[-0.33],[0.33]]),1,d-1) 
    vlimit_pulse = np.array([[-1],[1]])
    vlimit = np.hstack((vlimit_cc, vlimit_pulse)) # Velocity limits (2-d)

    np.random.seed(N * d * ger) # set seed
    for i in range(d):
        x[:,i] = np.matlib.repmat(xlimit[0,i],1,N) + (xlimit[1,i] - xlimit[0,i]) * np.random.rand(1,N)
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
            [t[iter,j], _, SoH[iter,j], _]= battery_pulse_charged(x[j]) # Battery simulation
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
            hsat = np.where(x[:,k] < xlimit[0,k], 1, 0)
            lsat = np.where(x[:,k] > xlimit[1,k], 2, 0)

            for j in range(N):
                if hsat[j] == 1:
                    x[j,k] = xlimit[0,k]
                elif lsat[j] == 2:
                    x[j,k] = xlimit[1,k]
    
        iter = iter + 1
        print(iter)

    return [SoH, t, ym, fym, fx, fxm]

if __name__ == '__main__':
    particle_swarm_optimization(20, 4, 50)