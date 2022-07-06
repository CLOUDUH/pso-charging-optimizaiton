'''
Author: CLOUDUH
Date: 2022-05-28 17:55:32
LastEditors: CLOUDUH
LastEditTime: 2022-07-05 16:49:13
Description: 
    Use coupling model which include battery 1-RC equivalent circuit model
    & thermal model & aging model.
'''

import sys
import time
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import numpy.matlib

VO_Tab = pd.read_csv("utils/VO.csv",header=None) # O not 0!!!
R0_Tab = pd.read_csv("utils/R0.csv",header=None)
R1_Tab = pd.read_csv("utils/R1.csv",header=None)
tau1_Tab = pd.read_csv("utils/tau1.csv",header=None)
Grid = pd.read_csv("utils/Grid.csv",header=None)

Qe = 3.3 # Battery Capacity (Ah)

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
    VO = griddata(Grid, VO_Tab, (SoC, temp), method='nearest')
    R0 = griddata(Grid, R0_Tab, (SoC, temp), method='nearest')
    tau1 = griddata(Grid, tau1_Tab, (SoC, temp), method='nearest')
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

    h = 40.106 # Heat transfer coefficient W/(m^2*K)
    A = 0.004317 # Heat exchange area at the cell surface m^2
    Tf = 298.15 # Homoeothermy
    m =0.0475 # Battery mass kg
    c = 800 # Specific heat capacity J/(kg*K)

    temp = Temp - 273.15
    R0 = griddata(Grid, R0_Tab, (SoC, temp), method='nearest')
    R0 = R0[0]
    R1 = griddata(Grid, R1_Tab, (SoC, temp), method='nearest')
    R1 = R1[0]
    dVO_T = (13.02 * SoC ** 5 - 38.54 * SoC ** 4 + 32.81 * SoC ** 3 + 2.042 \
        * SoC ** 2 - 13.93 * SoC + 4.8) / 1000
    dTemp = (I ** 2 * (R0 + R1) + I * Temp * dVO_T - h * A * (Temp - Tf)) / (m * c)
    Temp = Temp + dTemp * t_p

    return Temp

def aging_model(t_p:float, I:float, Temp:float, Qloss:float): #
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

    z = 0.55 # Order of Ah throughput
    E_a = 31700 # Activation energy for cycle aging J/mol
    R = 8.314 # Ideal gas constant J/(kg*K)
    alpha = 370.3 / Qe # Coefficient for aging acceleration caused by the current
    B = -47.836 * (I / Qe) ** 3 + 1215 * (I / Qe) ** 2 - 9418.9 * (I / Qe) + 36042

    try:
        b = (Qloss / (B * np.exp((-E_a + alpha * abs(I)) / (R * Temp)))) ** (1 - (1 / z))
    except:
        b = 0
    
    dQloss = (abs(I) / 3600) * z * B * np.exp((-E_a + alpha * abs(I)) / (R * Temp)) * b
    Qloss = Qloss + dQloss * t_p
    SoH = 1 - ((Qloss / Qe) / 0.2)

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

    return [V_t, SoC, Temp, Qloss, SoH]

def battery_charged(nCC:list): 
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
    SoC = 0.1
    Qloss = 0.001
    SoH = 1 - ((Qloss / Qe) / 0.2)
    Temp = 298.15
    i = 1
    E_ch = 0

    n = len(nCC)
    if n == 5:
        SoC_range = [0.2,0.4,0.6,0.8,0.999]
    elif n == 4:
        SoC_range = [0.25,0.5,0.75,0.999]
    elif n == 3:
        SoC_range = [0.4,0.8,0.999]
    elif n == 2:
        SoC_range = [0.7,0.999]
    elif n == 1:
        SoC_range = [0.999]
    else:
        raise ValueError("nCC list error")

    for j in range(n):

        while SoC < SoC_range[j]:

            if nCC[j] <= 0.01:
                break # Stop nonsence

            if (i-1) * t_p >= 4 * 3600:
                break # Stop falling

            [V_t, SoC] = equivalent_circuit_model(t_p, nCC[j], Temp, SoC) 
            Temp = thermal_model(t_p, nCC[j], Temp, SoC)
            [Qloss, SoH] = aging_model(t_p, nCC[j], Temp, Qloss)

            E_ch = E_ch + V_t * nCC[j] * t_p

            print(i, round(nCC[j],2), round(SoC, 2), round(Temp,2), round(Qloss,2), round(SoH,2))

            i = i + 1

    t = (i - 1) * 0.05

    return [SoH,t,E_ch]

def battery_pulse_charged(policy:list):
    '''
    Args: 
        policy: Charging policy [CC1, CC2, CC3, Pulse] (list) 
        thread: Thread number (int)
    Returns:
        t: Charging time (s)
        Q_loss: Capacity loss (Ah)
        SoH: State of health 
        Temp: Temperature (K)
    '''

    t_p = 1
    Temp = -10 + 273.15
    Q_loss = 0.001
    SoC = 0.01
    t = 0
    ratio_pulse = 0.2 # Duty ratio of pulse charging
    cycle_pulse = 10 # Cycle of the pulse charging
    flag = 0
    

    cur_cc1 = policy[0]
    cur_cc2 = policy[1]
    cur_cc3 = policy[2]
    cur_pulse = policy[3]
    iter = int(policy[4])
    thread = int(policy[5])

    while flag == 0 and SoC <= 0.3:
        [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, cur_cc1, SoC, Temp, Q_loss)
        t = t + 1
        if t >= 12 * 3600: # Timeout
            flag = 1 
            break
    time_cc1 = t
    
    while flag == 0 and SoC <= 0.6:
        [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, cur_cc2, SoC, Temp, Q_loss)
        t = t + 1
        if t >= 12 * 3600: # Timeout
            flag = 1 
            break
    time_cc2 = t - time_cc1
    
    while flag == 0 and SoC <= 0.9:
        [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, cur_cc3, SoC, Temp, Q_loss)
        t = t + 1
        if t >= 12 * 3600: # Timeout
            flag = 1 
            break
    time_cc3 = t - time_cc2
    
    while flag ==0 and SoC <= 0.999: # pulse charging
        t_start = t
        while t < t_start + cycle_pulse * ratio_pulse:
            [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, cur_pulse, SoC, Temp, Q_loss)
            t = t + 1
        while t < t_start + cycle_pulse * (1 - ratio_pulse):
            [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, 0, SoC, Temp, Q_loss)
            t = t + 1
        if t >= 12 * 3600: # Timeout
            flag = 1 
            break
    time_pulse = t - time_cc3

    t_cost = [t, time_cc1, time_cc2, time_cc3, time_pulse]

    policy_display =[round(policy[0],3), round(policy[1],3), round(policy[2],3), round(policy[3],3)]
    t_cost_display = [round(t_cost[0]/3600 ,2), round(t_cost[1]/3600,2), round(t_cost[2]/3600,2), round(t_cost[3]/3600,2), round(t_cost[4]/3600,2)]
    # print("Thread:", thread, "Policy:", policy_display, "Time Cost:", t_cost_display)

    print("Iter-Num:", iter, "-", thread, "\tSoH:",  round(100 * SoH, 3), "\tTemp:", 
        round(Temp, 3), "\tPly:", policy_display,"\tTime:", t_cost_display)

    return [t_cost, Q_loss, SoH, Temp, flag]

if __name__ == '__main__':
    
    # nCC = [2.0, 1.6, 1.2, 1.0, 0.8]
    # battery_charged(1, nCC)
    policy = [3.3, 3.3, 3.3, 10, 1, 1]
    # policy = [0.1, 0.1, 0.1, 0.1, 1, 16]
    print(battery_pulse_charged(policy))
 