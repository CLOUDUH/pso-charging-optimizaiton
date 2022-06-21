'''
Author: CLOUDUH
Date: 2022-05-28 17:55:32
LastEditors: CLOUDUH
LastEditTime: 2022-06-21 21:56:28
Description: 
    Use coupling model which include battery 1-RC equivalent circuit model
    & thermal model & aging model.
'''

import sys
from unittest import expectedFailure
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import numpy.matlib

VO_Tab = pd.read_csv("utils/VO.csv",header=None) # O not 0!!!
R0_Tab = pd.read_csv("utils/R0.csv",header=None)
R1_Tab = pd.read_csv("utils/R1.csv",header=None)
tau1_Tab = pd.read_csv("utils/tau1.csv",header=None)
Grid = pd.read_csv("utils/Grid.csv",header=None)

Tf = 298.15 # Homoeothermy
Qe = 3.3 # Battery Capacity Ah
h = 40.106 # Heat transfer coefficient W/(m^2*K)
c = 800 # Specific heat capacity J/(kg*K)
A = 0.004317 # Heat exchange area at the cell surface m^2
m =0.0475 # Battery mass kg
z = 0.4 # Order of Ah throughput
B = 130 # Pre-exponential factor
E_a = 18461 # Activation energy for cycle aging J/mol
R = 8.314 # Ideal gas constant J/(kg*K)
alpha = 32 # Coefficient for aging acceleration caused by the current

def equivalent_circuit_model(t_p:float, I:float, Temp:float, SoC:float): 
    '''Battery 1-RC Equivalent Circuit Model
    Args:
        t_p: Step (s)
        I: Battery charge current(charging positive) (A)
        Temp: Battery temperature (K)
        SoC: State of charge
    Returns:
        Vt: Battery voltage
        SoC: State of charge
    '''

    temp = Temp - 273.15
    VO = griddata(Grid, VO_Tab, (SoC, temp), method='linear')
    R0 = griddata(Grid, R0_Tab, (SoC, temp), method='linear')
    tau1 = griddata(Grid, tau1_Tab, (SoC, temp), method='linear')
    Vt = VO + I * (1 - np.exp(t_p / tau1)) + I * R0
    Vt = Vt[0]
    SoC = SoC + t_p * I / (3600 * Qe)

    return [Vt, SoC]

def thermal_model(t_p:float, I:float, Temp:float, SoC:float): 
    '''Battery Thermal Model
    Args:
        t_p: Step (s)
        I: Battery current(Charging positive) (A)
        Temp: Battery temperature (K)
        SoC: State of charge
    Returns:
        Temp: Battery temperature
    '''

    temp = Temp - 273.15
    R0 = griddata(Grid, R0_Tab, (SoC, temp), method='linear')
    R0 = R0[0]
    R1 = griddata(Grid, R1_Tab, (SoC, temp), method='linear')
    R1 = R1[0]
    dVO_T = (13.02 * SoC ** 5 - 38.54 * SoC ** 4 + 32.81 * SoC ** 3 + 2.042 \
        * SoC ** 2 - 13.93 * SoC + 4.8) / 1000
    dTemp = (I ** 2 * (R0 + R1) + I * Temp * dVO_T - h * A * (Temp - Tf)) / (m * c)
    Temp = Temp + dTemp * t_p

    return Temp

def aging_model(t_p:float, I:float, Temp:float, Qloss:float): 
    '''Battery Aging Model
    Args:
        t_p: Step (s)
        I: Battery current(charging positive) (A)
        Temp: Battery temperature (K)
        Qloss: Loss battery capacity (Ah)
    Returns:
        Qloss: Loss battery capacity (Ah)
        SoH: State of health 
    '''

    try:
        b = (Qloss / (B * np.exp((-E_a + alpha * abs(I)) / (R * Temp)))) ** (1 - (1 / z))
    except:
        b = 0

    dQloss = (abs(I) / 3600) * z * B * np.exp((-E_a + alpha * abs(I)) / (R * Temp)) * b
    Qloss = Qloss + dQloss * t_p
    SoH = 1 - ((Qloss / Qe) / 0.2)

    # print(dQloss, Qloss, SoH)
    return [Qloss, SoH]

def battery_model(t_p:float, I:float, SoC:float, Temp:float, Qloss:float):
    '''Battery Charging Model
    Args:
        t_p: Step (s)
        I: Battery current(charging positive) (A)
        V_t: Battery terminal voltage (V)
        SoC: State of charge
        Temp: Battery temperature (K)
        Qloss: Loss battery capacity (Ah)
        SoH: State of health
    Returns:
        V_t: Battery terminal voltage (V)
        SoC: State of charge
        Temp: Battery temperature (K)
        Qloss: Loss battery capacity (Ah)
        SoH: State of health
    Detail:
        This function is single step model, requires a while loop outside
    '''

    [V_t, SoC] = equivalent_circuit_model(t_p, I, Temp, SoC) 
    Temp = thermal_model(t_p, I, Temp, SoC)
    [Qloss, SoH] = aging_model(t_p, I, Temp, Qloss)

    # print("Cur:", round(I, 2), "SoC:", round(SoC, 2), "Temp:", round(Temp,2), \
    #     "Qloss:", round(Qloss,2), "SoH:", round(SoH,2))

    return [V_t, SoC, Temp, Qloss]

def battery_charged(SoC_init:float, pwr_chrg:float, Temp:float): 
    '''Battery Charging Whole Process
    Args:
        t_p: Step (s)
        CC: Battery charging constant current (d-1 list) 
    Returns:
        SoH: Whole charging process SoH
        t: Charging time cost
        E_ch: Total battery charge energy
    Detail:
        Do not need while loop
    '''
    t_p = 1
    SoC = SoC_init
    t = 0
    V_t = 3.0
    cur_chrg = pwr_chrg / V_t

    while SoC < SoC_init + 0.2:

        [V_t, SoC] = equivalent_circuit_model(t_p, cur_chrg, Temp, SoC) 
        Temp = thermal_model(t_p, cur_chrg, Temp, SoC)

        print(t, round(cur_chrg,2), round(SoC, 2), round(Temp,2))

        t = t + 1

    t = t / 3600

    return [t, V_t, Temp, cur_chrg]

if __name__ == '__main__':
    '''test process
    you can run this file directly and check
    '''
    
    nCC = [2.0, 1.6, 1.2, 1.0, 0.8]
    battery_charged(1, nCC)
    
 