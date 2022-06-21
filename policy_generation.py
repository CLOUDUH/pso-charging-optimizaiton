'''
Author: CLOUDUH
Date: 2022-06-20 15:33:01
LastEditors: CLOUDUH
LastEditTime: 2022-06-21 22:41:18
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

def charging_maximum_power_calc(t_s, t_e, date, latitude, Temp):
    '''
    Args:
        t_s: Start time (s)
        t_e: End time (s)
        date: Date (dd)
        latitude: Latitude (deg)
    Returns:
        pwr_mppt_list: List of MPPT power
    '''

    t_p = 1/3600
    t = 0
    volt_k0 = 20
    volt_k1 = 19
    pwr_k0 = 0
    pwr_k1 = 0
    volt_bat = 3.0
    Temp = 298.15
    volt_bat = 3.0
    pwr_chrg = 0
    SoC = 0.1
    Qloss = 0.001

    pwr_mppt_list = []
    t_list = []

    while t <= t_e:
        rad = irradiation_cal(t, date, latitude)
        [cur, pwr_k1, pwr_mp] = photovoltaic_model(rad, Temp, volt_k1)
        [volt_k1, volt_k0, cur_bat] = mppt_cal(volt_k0, volt_k1, pwr_k0, pwr_k1, volt_bat)
        pwr_k0 = pwr_k1

        pwr_load = load_model(t_p, t)
        
        if abs(t - t_s) <= 0.01:
            pwr_chrg = pwr_k1 - pwr_load
        
        if abs(SoC - 0.3) <= 0.01 or abs(SoC - 0.5) <= 0.01 or abs(SoC - 0.7) <= 0.01:
            pwr_chrg = pwr_k1 - pwr_load

        if pwr_chrg > pwr_k1:
            print("error")

        if SoC >= 0.9: break

        cur_bat = pwr_chrg / volt_bat
        
        [volt_bat, SoC, Temp, Qloss] = battery_model(t_p, cur_bat, SoC, Temp, Qloss)

        pwr_mppt_list.append(pwr_chrg)
        t_list.append(t)

        t = t + t_p
        
    return [t_list, pwr_mppt_list]

def policy_generation(t_0:float, factor:list, Q_e:float, date:int, latitude:float, Temp:float):
    '''
    Args:
        t_s: Start time (s)
        factor: List of factors
        date: Date (dd)
        latitude: Latitude (deg)
        Temp: Temperature (K)
    Returns:
        policy: List of charging policy(5-D)
    '''

    volt_k1 = 19
    t = 0
    t_list = []
    pwr_mp_list = []

    while t <= 24:
        rad = irradiation_cal(t, date, latitude)
        [_, _, pwr_mp] = photovoltaic_model(rad, Temp, volt_k1)
        pwr_mp_list.append(pwr_mp)
        t_list.append(t)
        t = t + 1/3600

    pwr_chrg_1 = np.interp(t_0, t_list, pwr_mp_list)
    [t, V_t, Temp, cc1] = battery_charged(0, pwr_chrg_1, Temp) # 充电过程仿真
    t_1 = t_0 + t # 记录第一阶段时间
    if t_1 - t_0 > 10 * 0.4: return 0 # 如果时间超限就直接返回错误
    if cc1 < 0.1 * Q_e or cc1 > 1 * Q_e: return 0 # 如果电流超限就直接返回错误
    
    pwr_chrg_2 = np.interp(t_1, t_list, pwr_mp_list)
    if  pwr_chrg_2 < 0.1 * Q_e or pwr_chrg_2 > 1 * Q_e: return 0
    [t, V_t, Temp, cc2] = battery_charged(0.2, factor[0] * pwr_chrg_2, Temp)
    t_2 = t_1 + t
    if t_2 - t_0 > 10 * 0.6: return 0

    pwr_chrg_3 = np.interp(t_2, t_list, pwr_mp_list)
    if  pwr_chrg_3 < 0.1 * Q_e or pwr_chrg_3 > 1 * Q_e: return 0
    [t, V_t, Temp, cc3] = battery_charged(0.4, factor[1] * pwr_chrg_3, Temp)
    t_3 = t_2 + t
    if t_3 - t_0 > 10 * 0.8: return 0

    pwr_chrg_4 = np.interp(t_3, t_list, pwr_mp_list)
    if  pwr_chrg_4 < 0.1 * Q_e or pwr_chrg_4 > 1 * Q_e: return 0
    [t, V_t, Temp, cc4] = battery_charged(0.6, factor[2] * pwr_chrg_4, Temp)
    t_4 = t_3 + t
    if t_4 - t_0 > 10 * 1: return 0

    pwr_chrg_5 = np.interp(t_4, t_list, pwr_mp_list)
    if  pwr_chrg_5 < 0.1 * Q_e or pwr_chrg_5 > 1 * Q_e: return 0
    [t, V_t, Temp, cc5] = battery_charged(0.8, factor[3] * pwr_chrg_5, Temp)
    t_5 = t_4 + t

    if t_1+t_2+t_3+t_4+t_5 > 10: return 0
    if t_1+t_2+t_3+t_4+t_5 < 4: return 0

    policy = [cc1, cc2, cc3, cc4, cc5]

    return policy

if __name__ == '__main__':
    # [t_list, pwr_mppt_list] = charging_maximum_power_calc(10, 24, 180, 30, 298.15)
    # plt.plot(t_list, pwr_mppt_list)
    # plt.show()

    a = policy_generation(8, [1,1,1,1,1], 20, 180, 30, 298.15)
    print(a)