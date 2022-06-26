'''
Author: CLOUDUH
Date: 2022-06-22 20:52:04
LastEditors: CLOUDUH
LastEditTime: 2022-06-25 10:42:36
Description: 
'''

import numpy as np
from model_battery import battery_model

def battery_pulse_charged(policy:list, thread:int):
    '''
    Args: 
        policy: Charging policy [CC1, CC2, CC3, Pulse] (list) 
    Returns:
        t: Charging time (s)
        Q_loss: Capacity loss (Ah)
        SoH: State of health 
        Temp: Temperature (K)
    '''
    
    t_p = 1
    Temp = 298.15
    Q_loss = 0.001
    SoC = 0.01
    t = 0
    ratio_pulse = 0.2 # Duty ratio of pulse charging
    cycle_pulse = 10 # Cycle of the pulse charging

    cur_cc1 = policy[0]
    cur_cc2 = policy[1]
    cur_cc3 = policy[2]
    cur_pulse = policy[3]

    while SoC <= 0.3:
        [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, cur_cc1, SoC, Temp, Q_loss)
        t = t + 1
    print(thread, SoC, SoH, Temp)
    
    while SoC <= 0.6:
        [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, cur_cc2, SoC, Temp, Q_loss)
        t = t + 1
    print(thread, SoC, SoH, Temp)
    
    while SoC <= 0.9:
        [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, cur_cc3, SoC, Temp, Q_loss)
        t = t + 1
    print(thread, SoC, SoH, Temp)
    
    while SoC <= 0.999: # pulse charging
        t_start = t
        while t < t_start + cycle_pulse * ratio_pulse:
            [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, cur_pulse, SoC, Temp, Q_loss)
            t = t + 1
        while t < t_start + cycle_pulse * (1 - ratio_pulse):
            [V_t, SoC, Temp, Q_loss, SoH] = battery_model(t_p, 0, SoC, Temp, Q_loss)
            t = t + 1

    return [t, Q_loss, SoH, Temp]

if __name__ == '__main__':
    policy = [1, 0.6, 0.4, 3]
    battery_pulse_charged(policy)