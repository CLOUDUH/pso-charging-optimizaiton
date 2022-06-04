'''
Author: CLOUDUH
Date: 2022-05-28 17:55:32
LastEditors: CLOUDUH
LastEditTime: 2022-06-03 19:41:22
Description: 
'''

from battery_model import battery_charged
from battery_model import battery_model
from solar_energy import solar_cell
from solar_energy import irradiation_cal
from solar_energy import mppt_cal

import sys
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

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

    t = 8
    seng = 0
    volt_k0 = 20
    volt_k1 = 19
    pwr_k0 = 0
    pwr_k1 = 0
    volt_bat = 3.0

    SoC = 0.1
    Qloss = 0.001
    Qe = 3.3
    SoH = 1 - ((Qloss / Qe) / 0.2)
    Temp = 298.15

    i = 0
    T = []
    Pwr = []
    Volt = []
    MP = []

    while t <= 24:

        rad = irradiation_cal(t,60,30)
        [cur, pwr_k1, pwr_mp] = solar_cell(rad, 298.15, volt_k1)
        [volt_k1, volt_k0, cur_bat] = mppt_cal(volt_k0, volt_k1, pwr_k0, pwr_k1, volt_bat)

        pwr_k0 = pwr_k1
        [volt_bat, SoC, Temp, Qloss] = battery_model(t_p, cur_bat, SoC, Temp, Qloss)

        print("t:", round(t,2), "Cur:", round(cur_bat, 2), "SoC:", round(SoC, 2), \
            "Temp:", round(Temp,2), "Qloss:", round(Qloss,2))

        T.append(t)
        Pwr.append(pwr_k1)
        Volt.append(volt_k1)
        MP.append(pwr_mp)
        seng = seng + pwr_k1 * t_p

        t = t + t_p/3600

    plt.plot(T, Pwr)
    plt.plot(T, MP)
    plt.show()
    
    return seng

if __name__ == '__main__':
    
    E_m = optimization_test(t_p)



    # nCC = [1.65, 1, 0.5]
    # [SoH,t,E_ch] = battery_charged(t_p,nCC)
    # print(E_m,SoH,t,E_ch)
    # print(f(SoH, E_ch))

