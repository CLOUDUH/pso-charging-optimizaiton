'''
Author: CLOUDUH
Date: 2022-06-06 11:26:59
LastEditors: CLOUDUH
LastEditTime: 2022-06-26 16:02:03
Description: A simple load model
'''

import numpy as np
import matplotlib.pyplot as plt

def load_model(t:float):

    t_p = 1
    pwr_high = 30 # note this is the power of the load when UAV in high altitude
    pwr_low = 20 # note this is the power of the load when UAV in low altitude

    t_up = 10 # time to rise from low altitude to high altitude
    t_high = 12 # time to stay in high altitude
    t_down = 18 # time to fall from high altitude to low altitude
    t_low = 20 # time to stay in low altitude

    if t < t_up:
        pwr = pwr_low
    elif t_up <= t < t_high:
        pwr = pwr_low + (t - t_up) / (t_high - t_up) * (pwr_high - pwr_low) 
    elif t_high <= t < t_down:
        pwr = pwr_high
    elif t_down <= t < t_low:
        pwr = pwr_high - (t - t_down) / (t_low - t_down) * (pwr_high - pwr_low)  
    else:
        pwr = pwr_low

    return pwr

if __name__ == '__main__':
    t_p = 1.0
    t = 0.0
    T = []
    Pwr = []
    while t < 24:
        t = t + t_p / 3600
        pwr = load_model(t_p, t)
        T.append(t)
        Pwr.append(pwr)
        
    print(T, Pwr)
    plt.plot(T, Pwr)
    plt.show()
    