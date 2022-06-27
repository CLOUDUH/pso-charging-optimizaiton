'''
Author: CLOUDUH
Date: 2022-05-28 17:55:32
LastEditors: CLOUDUH
LastEditTime: 2022-06-27 22:13:25
Description: Battery charging optimization by PSO
    Battery charging optimization program.
    Optimization algorithm is particle swarm optimization
'''

from threading import Thread
from utils.return_thread import ReturnThread
from model_battery import battery_pulse_charged
from model_photovoltaic import photovoltaic_model
from model_photovoltaic import irradiation_cal
from model_mppt import mppt_cal
from model_load import load_model

import sys
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

def object_func(SoH:float, t:float, policy:list, beta:float, t_list:list, remain_cur_list:list):
    '''Object Function Calculate
    Args: 
        SoH: State of Health
        t: Total charging time (s)
        policy: Charging policy (list)
        beta: Weight coeffeicent 
    Returns:
        J: Cost
    Description:
        The transformation of the objective function is 
        determined by the weight coefficients.
        except before that, when the particle 
        exceeds the limit, it will become infinite
    '''

    t_m = 2*3600 # Empirical charging time
    Qe = 3.3 # Battery Capacity (Ah)
    ratio_pulse = 0.2 # Duty ratio of pulse charging

    J = beta * t / t_m + (1 - beta) * SoH 

    t1 = np.interp(policy[0], remain_cur_list, t_list)
    t2 = t1 + 0.2 * Qe / policy[0]
    t3 = t2 + 0.2 * Qe / policy[1]
    t4 = t3 + 0.2 * Qe / policy[2]
    t5 = t4 + 0.2 * Qe / (policy[3] * ratio_pulse)

    cur_remain_2 = np.interp(t2, t_list, remain_cur_list)
    cur_remain_3 = np.interp(t3, t_list, remain_cur_list)
    cur_remain_4 = np.interp(t4, t_list, remain_cur_list)
    cur_remain_5 = np.interp(t5, t_list, remain_cur_list)

    if cur_remain_2 < policy[0] or cur_remain_3 < policy[1] or \
        cur_remain_4 < policy[2] or cur_remain_5 < policy[3] * ratio_pulse:
        J = np.inf

    if t < 2 * 3600 or t > 10 * 3600: J = np.inf
    if SoH < 0: J = np.inf
    return J

def clac_remain_current(date:int, latitude:float, vol_bat:float):
    '''Calculate Remain Power and Current
    Args:
        date: Today 0-365 
        latitude: (°)
        vol_bat: Battery voltage (V)
    Returns:
        cur_remain_list: Remain current list (A)
    '''

    t = 0
    Temp = 298.15
    t_list = []
    cur_remain_list = []
    cur_solar_list = []

    while t <= 24:

        rad = irradiation_cal(t, date, latitude)
        [cur_solar, _, pwr_solar] = photovoltaic_model(rad, Temp, vol_bat)
        pwr_load = load_model(t)
        cur_load = pwr_load / vol_bat

        cur_remain = cur_solar - cur_load
        if cur_remain < 0: cur_remain = 0
        t = t + 1/3600
        
        cur_remain_list.append(cur_remain)
        t_list.append(t)

    # plt.plot(t_list, cur_remain_list)
    # plt.show()

    return [t_list, cur_remain_list]

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

    date = 180 # Today 0-365
    latitude = 30 # (°)
    vol_bat = 3.6 # Battery voltage (V)
    w = 0.729 # Inertial weight
    c1 = 1.49115 # Self learning factor
    c2 = 1.49115 # Swarm learning factor 
    beta = 1 # Weight coefficient 1: fastest; 0: healthiest
    ger = 50 # The maximum number of iterations  
    iter = 1  # Initial iteration
    np.random.seed(N * d * ger) # set seed

    # Initialize the particle position and velocity
    x = np.zeros((N,d)) # Particle Position (N-d)
    v = np.zeros((N,d)) # Particle Velcocity (N-d)

    [t, cur_remain] = clac_remain_current(date, latitude, vol_bat) # caclulate the remain current

    xlimit_cc = np.matlib.repmat(np.array([[0],[3.3]]),1,d-1) 
    xlimit_pulse = np.array([[0],[6.6]])
    xlimit = np.hstack((xlimit_cc, xlimit_pulse)) # Charging current limits (2-d)

    vlimit_cc = np.matlib.repmat(np.array([[-0.33],[0.33]]),1,d-1) 
    vlimit_pulse = np.array([[-1],[1]])
    vlimit = np.hstack((vlimit_cc, vlimit_pulse)) # Velocity limits (2-d)

    # Initialize the particle position
    for i in range(d):
        x[:,i] = np.matlib.repmat(xlimit[0,i],1,N) + (xlimit[1,i] - xlimit[0,i]) * np.random.rand(1,N)
        
    # Initialize the particle velocity
    for i in range(d):
        v[:,i] = np.matlib.repmat(vlimit[0,i],1,N) + (vlimit[1,i] - vlimit[0,i]) * np.random.rand(1,N)
        
    xm = x # The best known position of particle (N-d)
    ym = np.zeros((1,d)) # The best known position of entire swarm (1-d)
    SoH = np.zeros((ger,N)) # SoH of each iteration of each particle (ger-N)
    t = np.zeros((ger,N)) # Charging time of each iteration of each particle (ger-N)
    fx = np.zeros((ger,N)) # Objective function value of each iteration of each particle (ger-N)
    fxm = np.zeros((N,1)) # Optimal objective function value of particle (N-1)
    fym = float("-inf") # Global optimal objective function value

    while iter <= ger:

        threads = []

        for j in range(N): 
            exec(f'th{j} = ReturnThread(battery_pulse_charged, (x[{j}], {j},))')
            exec(f'threads.append(th{j})')
        
        for j in range(N): 
            threads[j].start() # Parallel computing

        for j in range(N):
            threads[j].join() # Wait all thread to finish

        for j in range(N): 
            [t[iter,j], _, SoH[iter,j], _]= threads[j].get_result() # get result of each thread
            fx[iter,j] = object_func(SoH[iter,j], t[iter,j], x[j], 0.5, t, cur_remain) # Optimal function value
            
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
            hsat = np.where(v[:,k] < vlimit[0,k], 1, 0) #
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
    [SoH, t, ym, fym, fx, fxm] = particle_swarm_optimization(20, 4, 50)

    # [t, cur_remain] = clac_remain_current(180, 30, 3.6)
