'''
Author: CLOUDUH
Date: 2022-05-28 17:55:32
LastEditors: CLOUDUH
LastEditTime: 2022-06-20 20:41:05
Description: 
'''

from json import load
from model_battery import battery_charged
from model_battery import battery_model
from model_photovoltaic import photovoltaic_model
from model_photovoltaic import irradiation_cal
from model_mppt import mppt_cal
from model_load import load_model

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

    t = 0
    
    egy_solar = 0
    egy_mppt = 0
    egy_load = 0
    egy_battery = 0
    egy_remain = 0

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
    t_list = []
    pwr_solar_list = []
    pwr_mppt_list = []
    pwr_load_list = []
    pwr_remain_list = []

    volt_list = []

    while t <= 24:

        rad = irradiation_cal(t,180,30)
        [cur, pwr_k1, pwr_mp] = photovoltaic_model(rad, 298.15, volt_k1)
        [volt_k1, volt_k0, cur_bat] = mppt_cal(volt_k0, volt_k1, pwr_k0, pwr_k1, volt_bat)
        pwr_k0 = pwr_k1

        # [volt_bat, SoC, Temp, Qloss] = battery_model(t_p, cur_bat, SoC, Temp, Qloss)

        pwr_load = load_model(t_p, t)
        pwr_remain = pwr_k1 - pwr_load

        print("t:", round(t,2), "Cur:", round(cur_bat, 2), "SoC:", round(SoC, 2), \
            "Temp:", round(Temp,2), "Qloss:", round(Qloss,2))

        t_list.append(t)
        pwr_solar_list.append(pwr_k1)
        pwr_mppt_list.append(pwr_mp)
        pwr_load_list.append(pwr_load)
        pwr_remain_list.append(pwr_remain)

        volt_list.append(volt_k1)
        egy_solar = egy_solar + pwr_k1 * t_p
        egy_mppt = egy_mppt + pwr_mp * t_p
        egy_load = egy_load + pwr_load * t_p
        egy_remain = egy_remain + pwr_remain * t_p

        t = t + t_p/3600

    plt.plot(t_list, pwr_solar_list)
    plt.plot(t_list, pwr_mppt_list)
    plt.plot(t_list, pwr_load_list)
    plt.plot(t_list, pwr_remain_list)
    plt.show()

    print("Egy_solar:", round(egy_solar,2), "Egy_mppt:", round(egy_mppt,2), \
        "Egy_load:", round(egy_load,2), "Egy_remain:", round(egy_remain,2))
    
    return None

if __name__ == '__main__':
    
    optimization_test(t_p)

    # nCC = [1.65, 1, 0.5]
    # [SoH,t,E_ch] = battery_charged(t_p,nCC)
    # print(E_m,SoH,t,E_ch)
    # print(f(SoH, E_ch))

