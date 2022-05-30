'''
Author: CLOUDUH
Date: 2022-05-28 17:55:32
LastEditors: CLOUDUH
LastEditTime: 2022-05-30 19:05:35
Description: 
'''

from battery_model import battery_charged
from battery_model import battery_model
from solar_energy import solar_cell
from solar_energy import irradiation_cal

import sys
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import numpy.matlib

t_p = 1 # Step (s)
t_m = 2*3600 # Empirical charging time

f = lambda SoH, E_ch: (1 - SoH) + ((E_m - E_ch) / E_m)   # Object function

def optimization_test(t_p:float):
    '''optimization_test
    Args:
        t_p: Step (s)
    Returns:
        normal: CC list in paper
    '''

    t = 0
    seng = 0

    while t <= 24:
        ird = irradiation_cal(t,60,30)
        [scur, spwr] = solar_cell(ird,298.15,30)
        # bcur = spwr / 
        # battery_model(t_p,bcur,)
        seng = seng + spwr * t_p
        t = t + t_p/3600
    
    return seng

if __name__ == '__main__':
    
    E_m = optimization_test(t_p) 
    nCC = [1.65, 1, 0.5]
    [SoH,t,E_ch] = battery_charged(t_p,nCC)
    print(E_m,SoH,t,E_ch)
    print(f(SoH, E_ch))