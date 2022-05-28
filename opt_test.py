'''
Author: CLOUDUH
Date: 2022-05-28 17:55:32
LastEditors: CLOUDUH
LastEditTime: 2022-05-28 22:26:53
Description: 
'''

from battery_model import battery_charging
from solar_energy import solar_cell
from solar_energy import irradiation_cal

import sys
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import numpy.matlib

t_p = 1 # Step (s)
t_m = 2*3600 # Empirical charging time

def optimization_test(t_p:float):
    '''optimization_test
    Args:
        t_p: Step (s)
    Returns:
        normal: CC list in paper
    '''

    # f = lambda SoH, E_ch: (1 - SoH) + ((E_m - E_ch) / E_m)   # Object function

    t = 0
    seng = 0

    while t <= 24:
        ird = irradiation_cal(t,60,30)
        [scur, spwr] = solar_cell(ird,298.15,30)
        seng = seng + spwr * t_p
        t = t + t_p/3600
    
    return seng

if __name__ == '__main__':
    
    E_m = optimization_test(t_p) 
    nCC = [1.65, 1, 0.5]
    [SoH,t,E_ch] = battery_charging(t_p,nCC)
    print(E_m,SoH,t,E_ch)