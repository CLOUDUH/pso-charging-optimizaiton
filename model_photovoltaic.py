'''
Author: CLOUDUH
Date: 2022-05-28 17:55:32
LastEditors: CLOUDUH
LastEditTime: 2022-07-16 14:41:25
Description: 
    Solar energy calculate
    - Solar irradiation calculate
    - Solar cell calculate
    - Maybe MPPT?
'''

import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from func.sat import sat

def irradiation_cal(t:float, date:float, latitude:float):
    '''Solar irradiation calculate
    Args:
        t: Time (h) 0-24
        date: Today 0-365 
        latitude: (°)
    Returns:
        irradiation: Unit area intensity of solar radiation (W/m^2)
    '''

    I_SC = 1367.0 # Extra-solar intensity of solar radiation
    r_0 = 149597890.0 # Average distance of the eart
    epislon = 1/298.256 # Oblateness of the earth
    tau = 1.0 # Projection coefficient

    alpha_day = 2 * 3.1415926 * (date - 4) / 365 # Number of days Angle
    r = r_0 * (1 - epislon ** 2) / (1 + epislon * np.cos(alpha_day)) # Actual distance of the eart
    I_0n = I_SC * (r_0 / r) ** 2 # Extra-solar intensity of solar radiation
    omega = 3.1415926 - (3.1415926 * t / 12) # Hour angle
    delta = (23.45 * 3.1415926 / 180) * np.sin(360 * (284 + date) * 3.1415926 / 365 * 180) # Solar elevation angle
    theta = 3.1415926 / 2 - np.arccos(np.sin(latitude * 3.1415926 / 180) \
        * np.sin(delta) + np.cos(latitude * 3.1415926 / 180) * np.cos(delta) * np.cos(omega)) # Zenith angle

    irradiation = max(I_0n * tau * np.sin(theta), 0) # Unit area intensity of solar radiation

    return irradiation

def photovoltaic_model(irradiation:float, Temp:float, volt:float):
    '''Solar cell engineering model
    Args:
        irradiation: Unit area intensity of solar radiation (W/m^2)
        Temp: Solar cell temperature (K)
        volt: Terminal voltage (V)
    Returns:
        cur: Solar cell current (A)
        pwr: Current power (W)
        pwr_mp: Maximum power (W)
    '''

    vol_oc_ref = 12 # Open circuit voltage
    vol_mp_ref = 10 # Maximum power point voltage
    cur_sc_ref = 1.2 # Short circuit current
    cur_mp_ref = 1 # Maximum power point current

    volt_oc = vol_oc_ref * np.log(np.exp(1) + 0.0005 * (abs(irradiation - 1000))) * (1 - 0.00288 * (abs(Temp - 298.15)))
    volt_mp = vol_mp_ref * np.log(np.exp(1) + 0.0005 * (abs(irradiation - 1000))) * (1 - 0.00288 * (abs(Temp - 298.15)))
    cur_sc = cur_sc_ref * (irradiation / 1000) * (1 + 0.0025 * (abs(Temp - 298.15)))
    cur_mp = cur_mp_ref * (irradiation / 1000) * (1 + 0.0025 * (abs(Temp - 298.15)))

    try:
        c2 = ((volt_mp / volt_oc) - 1) * (np.log(1 - (cur_mp / cur_sc))) ** -1
        c1 = (1 - (cur_mp / cur_sc)) * np.exp(-volt_mp / (c2 * volt_oc))
        cur = cur_sc *(1-c1 *(np.exp( volt / (c2 * volt_oc)) - 1))
    except:
        cur = 0

    cur = sat(cur, 0, math.inf)
    cur = cur_mp * volt_mp / volt
    
    pwr = cur * volt
    pwr_mp = cur_mp * volt_mp # maximum power

    return [cur, pwr, pwr_mp]

def clac_remain_current(t_p:float, date:int, latitude:float):
    '''Calculate Remain Power and Current
    Args:
        date: Today 0-365 
        latitude: (°)
        vol_bat: Battery voltage (V)
    Returns:
        t_pv_list: Time list (s)
        cur_remain_list: Remain current list (A)
    '''

    t = 0
    Temp = 298.15
    vol_bat = 4.2 # Maximum battery voltage
    egy_solar = 0
    t_list = [0]
    cur_list = [0]
    pwr_list = [0]
    egy_list = [0]

    while t <= 24 * 3600:

        rad = irradiation_cal(t/3600, date, latitude)
        [cur_solar, pwr_solar, _] = photovoltaic_model(rad, Temp, vol_bat)
       
        if pwr_solar < 0: pwr_solar = 0

        if pwr_list[-1] == 0 and pwr_solar > 0:
            t_riseup = t
        elif pwr_list[-1] > 0 and pwr_solar == 0:
            t_falldown = t
        else:
            pass
        
        egy_solar = egy_solar + pwr_solar * t_p / 3600

        t = t + t_p
        
        cur_list.append(cur_solar)
        pwr_list.append(pwr_solar)
        egy_list.append(egy_solar)
        t_list.append(t)

    print("Power(W):", round(max(pwr_list),2), "Energy(Wh):", round(egy_list[-1],2), "Time(h):", round(t_riseup/3600,2), round(t_falldown/3600,2))

    return [t_list, cur_list, pwr_list, egy_list, t_riseup, t_falldown]

if __name__ == '__main__':

    [t_pv, cur_list, pwr_pv, egy_pv, t_riseup, t_falldown] = clac_remain_current(0.1, 180, 30)
    plt.plot(t_pv, pwr_pv)
    plt.plot(t_pv, egy_pv)
    plt.show()

    