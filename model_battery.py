'''
Author: CLOUDUH
Date: 2022-05-28 17:55:32
LastEditors: CLOUDUH
LastEditTime: 2022-07-07 22:24:09
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
import matplotlib.pyplot as plt

from func.sciplt import sciplt_battery

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
    time_cc3 = t - time_cc2 - time_cc1
    
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
    time_pulse = t - time_cc3 - time_cc2 - time_cc1

    t_cost = [t, time_cc1, time_cc2, time_cc3, time_pulse]

    policy_display =[round(policy[0],3), round(policy[1],3), round(policy[2],3), round(policy[3],3)]
    t_cost_display = [round(t_cost[0]/3600 ,2), round(t_cost[1]/3600,2), round(t_cost[2]/3600,2), round(t_cost[3]/3600,2), round(t_cost[4]/3600,2)]
    # print("Thread:", thread, "Policy:", policy_display, "Time Cost:", t_cost_display)

    print("Iter-Num:", iter, "-", thread, "\tSoH:",  round(100 * SoH, 3), "\tTemp:", 
        round(Temp, 3), "\tPly:", policy_display,"\tTime:", t_cost_display)

    return [t_cost, Q_loss, SoH, Temp, flag]

def pulse_charged_plot(policy:list):

    t_p = 1
    Temp = 25 + 273.15
    Q_loss = 0.001
    SoC = 0.01
    t = 0
    ratio_pulse = 0.2 # Duty ratio of pulse charging
    cycle_pulse = 10 # Cycle of the pulse charging

    cur_cc1 = policy[0]
    cur_cc2 = policy[1]
    cur_cc3 = policy[2]
    cur_pulse = policy[3]

    t_list = []
    SoH_list = []
    Temp_list = []
    Q_loss_list = []
    V_t_list = []

    while SoC <= 0.3:
        [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, cur_cc1, SoC, Temp, Q_loss)
        t_list.append(t)
        SoH_list.append(SoH)
        Temp_list.append(Temp)
        Q_loss_list.append(Q_loss)
        V_t_list.append(V_t)
        t = t + 1
    
    while SoC <= 0.6:
        [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, cur_cc2, SoC, Temp, Q_loss)
        t_list.append(t)
        SoH_list.append(SoH)
        Temp_list.append(Temp)
        Q_loss_list.append(Q_loss)
        V_t_list.append(V_t)
        t = t + 1
    
    while SoC <= 0.9:
        [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, cur_cc3, SoC, Temp, Q_loss)
        t_list.append(t)
        SoH_list.append(SoH)
        Temp_list.append(Temp)
        Q_loss_list.append(Q_loss)
        V_t_list.append(V_t)
        t = t + 1
    
    while SoC <= 0.999: # pulse charging
        t_start = t
        while t < t_start + cycle_pulse * ratio_pulse:
            [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, cur_pulse, SoC, Temp, Q_loss)
            t_list.append(t)
            SoH_list.append(SoH)
            Temp_list.append(Temp)
            Q_loss_list.append(Q_loss)
            V_t_list.append(V_t)
            t = t + 1

        while t < t_start + cycle_pulse * (1 - ratio_pulse):
            [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, 0, SoC, Temp, Q_loss)
            t_list.append(t)
            SoH_list.append(SoH)
            Temp_list.append(Temp)
            Q_loss_list.append(Q_loss)
            V_t_list.append(V_t)
            t = t + 1

    return [t_list, SoH_list, Temp_list, Q_loss_list, V_t_list]

def battery_cccv_charged(): 

    t_p = 1
    SoC = 0.1
    Q_loss = 0.001
    Temp = 298.15
    t = 0

    cc = 0.5 * 3.3
    cv = 4.2

    t_list = []
    SoH_list = []
    Temp_list = []
    Q_loss_list = []
    V_t_list = []

    while SoC <= 1.0:

        [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, cc, SoC, Temp, Q_loss)
        t_list.append(t)
        SoH_list.append(SoH)
        Temp_list.append(Temp)
        Q_loss_list.append(Q_loss)
        V_t_list.append(V_t)
        t = t + 1

    return [t_list, SoH_list, Temp_list, Q_loss_list, V_t_list]

if __name__ == '__main__':

    policy = [0.98703503, 1.70897805, 2.49398114, 6.6]

    [t1_list, SoH1_list, Temp1_list, Q_loss1_list, V_t1_list] = pulse_charged_plot(policy)

    [t2_list, SoH2_list, Temp2_list, Q_loss2_list, V_t2_list] = battery_cccv_charged()

    y1 = {'SoH': SoH1_list, 'Temp': Temp1_list, 'Q_loss': Q_loss1_list, 'V_t': V_t1_list}

    y2 = {'SoH': SoH2_list, 'Temp': Temp2_list, 'Q_loss': Q_loss2_list, 'V_t': V_t2_list}

    plt.subplot(221)
    sciplt_battery([t1_list,t2_list], [V_t1_list, V_t2_list], "Time(s)","Voltage(V)","Terminal Voltage",["PSO","CCCV"])
    plt.subplot(222)
    sciplt_battery([t1_list,t2_list], [SoH1_list, SoH2_list], "Time(s)","SoH(%)","SoH",["PSO","CCCV"])
    plt.subplot(223)
    sciplt_battery([t1_list,t2_list], [Temp1_list, Temp2_list], "Time(s)","Temperature(K)","Temperature",["PSO","CCCV"])
    plt.subplot(224)
    sciplt_battery([t1_list,t2_list], [Q_loss1_list, Q_loss2_list], "Time(s)","Q loss(Ah)","Capacity Loss",["PSO","CCCV"])
    plt.show()

    