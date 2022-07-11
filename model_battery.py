'''
Author: CLOUDUH
Date: 2022-05-28 17:55:32
LastEditors: CLOUDUH
LastEditTime: 2022-07-11 15:55:50
Description: 
    Use coupling model which include battery 1-RC equivalent circuit model
    & thermal model & aging model.
'''

import numpy as np
from scipy.interpolate import interp2d
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt

SOC_ARRAY = np.arange(0, 1, 0.01)
TEMP_ARRAY = [-10+273.15, 0+273.15, 25+273.15, 40+273.15, 60+273.15]
OCV_TABLE = pd.read_csv("utils/ocv_table.csv",header=None)
R0_TABLE = pd.read_csv("utils/r0_table.csv",header=None)
R1_TABLE = pd.read_csv("utils/r1_table.csv",header=None)
TAU1_TABLE = pd.read_csv("utils/tau1_table.csv",header=None)

ocv_lookup = interp2d(TEMP_ARRAY, SOC_ARRAY, OCV_TABLE, kind='linear')
r0_lookup = interp2d(TEMP_ARRAY, SOC_ARRAY, R0_TABLE, kind='linear')
r1_lookup = interp2d(TEMP_ARRAY, SOC_ARRAY, R1_TABLE, kind='linear')
tau1_lookup = interp2d(TEMP_ARRAY, SOC_ARRAY, TAU1_TABLE, kind='linear')

cap_original = 3.3 # Battery Capacity (Ah)

def equivalent_circuit_model(t_p:float, cur:float, temp:float, soc:float): 
    '''Battery 1-RC Equivalent Circuit Model
    Args:
        t_p: Step (s)
        cur: Battery charge current(charging positive) (A)
        temp: Battery temperature (K)
        soc: State of charge
    Returns:
        volt: Battery voltage
        soc: State of charge
    '''

    ocv = ocv_lookup(temp, soc)[0]
    r0 = r0_lookup(temp, soc)[0]
    tau1 = tau1_lookup(temp, soc)[0]

    volt = ocv + cur * (1 - np.exp(t_p / tau1)) + cur * r0
    soc = soc + t_p * cur / (3600 * cap_original)

    return [volt, soc]

def thermal_model(t_p:float, cur:float, temp:float, soc:float): 
    '''Battery Thermal Model
    Args:
        t_p: Step (s)
        cur: Battery current(Charging positive) (A)
        temp: Battery temperature (K)
        soc: State of charge
    Returns:
        temp: Battery temperature
    '''

    h = 40.106 # Heat transfer coefficient W/(m^2*K)
    A = 0.004317 # Heat exchange area at the cell surface m^2
    Tf = 298.15 # Homoeothermy
    m =0.0475 # Battery mass kg
    c = 800 # Specific heat capacity J/(kg*K)

    r0 = r0_lookup(temp, soc)[0]
    r1 = r1_lookup(temp, soc)[0]

    entropy_coef = (13.02 * soc ** 5 - 38.54 * soc ** 4 + 32.81 * soc ** 3 + 2.042 \
        * soc ** 2 - 13.93 * soc + 4.8) / 1000
    
    dtemp = (cur ** 2 * (r0 + r1) + cur * temp * entropy_coef - h * A * (temp - Tf)) / (m * c)
    
    temp = temp + dtemp * t_p

    return temp

def aging_model(t_p:float, cur:float, temp:float, cap:float): 
    '''Battery Aging Model
    Args:
        t_p: Step (s)
        I: Battery current(charging positive) (A)
        Temp: Battery temperature (K)
        cap: Battery capacity (Ah)
    Returns:
        cap: Battery capacity (Ah)
        cap_loss: Battery capacity loss (Ah)
        soh: State of health (0-1)
    '''

    cap_loss = cap_original - cap # Battery capacity loss (Ah)

    z = 0.55 # Order of Ah throughput
    E = 31700 # Activation energy for cycle aging J/mol
    R = 8.314 # Ideal gas constant J/(kg*K)
    alpha = 370.3 / cap_original # Coefficient for aging acceleration caused by the current
    B = -47.836 * (cur / cap_original) ** 3 + 1215 * (cur / cap_original) ** 2 - 9418.9 * (cur / cap_original) + 36042

    try:
        b = (cap_loss / (B * np.exp((-E + alpha * abs(cur)) / (R * temp)))) ** (1 - (1 / z))
    except:
        b = 0
    
    dloss = (abs(cur) / 3600) * z * B * np.exp((-E + alpha * abs(cur)) / (R * temp)) * b

    cap_loss = cap_loss + dloss * t_p
    cap = cap_original - cap_loss

    soh = 1 - ((cap_loss / cap_original) / 0.2)

    return [cap, cap_loss ,soh]

def battery_model(t_p:float, cur:float, soc:float, temp:float, cap:float):
    '''Battery Charging Model
    Args:
        t_p: Step (s)
        I: Battery current(charging positive) (A)
        SoC: State of charge
        Temp: Battery temperature (K)
    Returns:
        volt: Battery terminal voltage (V)
        soc: State of charge
        temp: Battery temperature (K)
        cap: Battery capacity (Ah)
        cap_loss: Battery capacity loss (Ah)
        soh: State of health
    Detail:
        This function is single step model, requires a while loop outside
    '''

    [volt, soc] = equivalent_circuit_model(t_p, cur, temp, soc)
    temp = thermal_model(t_p, cur, temp, soc)
    [cap, cap_loss ,soh] = aging_model(t_p, cur, temp, cap)
    
    # print("Cur:", round(cur, 2), "SoC:", round(soc, 2), "Temp:", round(temp,2), "Qloss:", round(cap,2), "SoH:", round(soh,2))

    return [volt, soc, temp, cap, cap_loss, soh]

if __name__ == '__main__':

    policy1 = [0.98703503, 1.70897805, 2.49398114, 6.6]
    policy2 = [0.90453935, 1.68994086, 2.53827688, 4.13408792]

