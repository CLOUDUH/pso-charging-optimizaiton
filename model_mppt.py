'''
Author: CLOUDUH
Date: 2022-06-06 11:17:32
LastEditors: CLOUDUH
LastEditTime: 2022-07-07 21:21:33
Description: 
'''

import math
import matplotlib.pyplot as plt
from func.sat import sat

from model_photovoltaic import photovoltaic_model
from model_photovoltaic import irradiation_cal

def mppt_cal(volt_k0:float, volt_k1:float, pwr_k0:float, pwr_k1:float, volt_out:float):
    '''Solar Cell Maximum Power Point Tracker
    Args: 
        volt_d: voltage step
        volt_k0: k-2 volt
        volt_k1: k-1 volt
        pwr_k0: k-1 power
        pwr_k1: k power
        volt_out: MPPT output voltage
    Returns: 
        volt_k0: k-1 volt 
        volt_k1: k volt
        cur_out: Battery current
    Detail: 
    '''

    pwr_tlr = 1e-6 # MPPT power tolerance
    volt_tlr = 1e-6 # MPPT voltage tolerance
    volt_d = 1e-2 # MPPT voltage distance


    if pwr_k1 - pwr_k0 > pwr_tlr :
        if volt_k1 - volt_k0 > volt_tlr:
            volt_in = volt_k1 + volt_d
        elif volt_k1 - volt_k0 < -volt_tlr:
            volt_in = volt_k1 - volt_d
        else:
            volt_in = volt_k1
    if pwr_k1 - pwr_k0 < pwr_tlr :
        if volt_k1 - volt_k0 > volt_tlr:
            volt_in = volt_k1 - volt_d
        elif volt_k1 - volt_k0 < -volt_tlr:
            volt_in = volt_k1 + volt_d
        else:
            volt_in = volt_k1

    volt_in = sat(volt_in, 0, math.inf)
    volt_k0 = volt_k1
    volt_k1 = volt_in

    try:
        cur_out = sat(pwr_k1/volt_out, 0, math.inf)
    except:
        cur_out = 0

    return [volt_k1, volt_k0, cur_out]

if __name__ == '__main__':
    print(sat(10, -1, 2))
    print(sat(10, -1, math.inf))

    t = 0
    volt_k0 = 20
    volt_k1 = volt_k0 - 0.001
    pwr_k0 = 0
    pwr_k1 = 0
    volt_bat = 30
    i = 0
    T = []
    Pwr = []
    Volt = []
    MP = []

    while t <= 24:
        rad = irradiation_cal(t, 60, 30)
        
        [cur, pwr_k1, pwr_mp] = photovoltaic_model(rad, 298.15, volt_k1)
        [volt_k1, volt_k0, cur_bat] = mppt_cal(volt_k0, volt_k1, pwr_k0, pwr_k1, volt_bat)
        pwr_k0 = pwr_k1

        T.append(t)
        Pwr.append(pwr_k1)
        Volt.append(volt_k1)
        MP.append(pwr_mp)

        t = t + 1 / 3600

    plt.plot(T, Pwr)
    plt.plot(T, MP)
    # plt.plot(T, Volt)
    plt.show()