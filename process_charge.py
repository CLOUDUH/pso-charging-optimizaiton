'''
Author: CLOUDUH
Date: 2022-07-09 14:58:26
LastEditors: CLOUDUH
LastEditTime: 2022-07-16 16:50:31
Description: 
'''

import matplotlib.pyplot as plt
import numpy as np

from model_battery import battery_model

def cc_charge(t_p:float, cur:float, bdy:list, flag_timeout:int, data_log:dict):
    '''Battery Constant Current Charge Process
    Args:
        t_p: Pulse time (s)
        cur: Pulse charge current (A)
        bdy: Pulse charge SoC boundaries list
        flag_timeout: Timeout flag (0: no timeout, 1: timeout)
        data_log: Battery data log (dict)
    Return:
        data_log: Battery data log (dict)
        t_charge: Charging time (s)
        flag_timeout: Timeout flag (0: no timeout, 1: timeout)
    '''

    t_start = data_log['t'][-1]
    soc = data_log['soc'][-1]
    volt = data_log['volt'][-1]
    egy = data_log['egy'][-1]
    volt_tau1 = data_log['volt_tau1'][-1]
    temp = data_log['temp'][-1]
    cap = data_log['cap'][-1]
    
    t = t_start

    while soc >= bdy[0] and soc <=bdy[1]:
        [soc, volt, pwr, volt_tau1, temp, cap, cap_loss, soh] = battery_model(t_p, cur, soc, volt_tau1, temp, cap)
        egy = egy + pwr * t_p / 3600

        data_log['t'].append(t)
        data_log['soc'].append(soc)
        data_log['volt'].append(volt)
        data_log['cur'].append(cur)
        data_log['pwr'].append(pwr)
        data_log['egy'].append(egy)
        data_log['volt_tau1'].append(volt_tau1)
        data_log['temp'].append(temp)
        data_log['cap'].append(cap)
        data_log['cap_loss'].append(cap_loss)
        data_log['soh'].append(soh)
        
        t = t + t_p

        if t >= 12 * 3600: # Timeout    
            flag_timeout = 1 
            break
    
    t_charge = data_log['t'][-1] - t_start

    return [t_charge, flag_timeout, data_log]

def cv_charge(t_p:float, cv:float, bdy:list, flag_timeout:int, data_log:dict):
    '''Battery Constant Voltage Charge Process
    Args:
        t_p: Pulse time (s)
        cur: Pulse charge current (A)
        bdy: Pulse charge SoC boundaries list
        flag_timeout: Timeout flag (0: no timeout, 1: timeout)
        data_log: Battery data log (dict)
    Return:
        data_log: Battery data log (dict)
        t_charge: Charging time (s)
        flag_timeout: Timeout flag (0: no timeout, 1: timeout)
    '''

    t_start = data_log['t'][-1]
    soc = data_log['soc'][-1]
    volt = data_log['volt'][-1]
    egy = data_log['egy'][-1]
    volt_tau1 = data_log['volt_tau1'][-1]
    temp = data_log['temp'][-1]
    cap = data_log['cap'][-1]

    t = t_start
    cur = 1.65

    while soc > bdy[0] and soc <=bdy[1]:

        if volt > cv:
            cur = cur - 1e-2
        else:
            pass

        if cur <= 6.5e-2: break

        [soc, volt, pwr, volt_tau1, temp, cap, cap_loss, soh] = battery_model(t_p, cur, soc, volt_tau1, temp, cap)
        egy = egy + pwr * t_p / 3600

        data_log['t'].append(t)
        data_log['soc'].append(soc)
        data_log['volt'].append(volt)
        data_log['cur'].append(cur)
        data_log['pwr'].append(pwr)
        data_log['egy'].append(egy)
        data_log['volt_tau1'].append(volt_tau1)
        data_log['temp'].append(temp)
        data_log['cap'].append(cap)
        data_log['cap_loss'].append(cap_loss)
        data_log['soh'].append(soh)

        if t >= 12 * 3600: # Timeout    
            flag_timeout = 1 
            break
    
    t_charge = data_log['t'][-1] - t_start

    return [t_charge, flag_timeout, data_log]

def pulse_charge(t_p:float, cur:float, ratio:float, cycle:float, bdy:list, flag_timeout:int, data_log:dict):
    '''Battery Pulse Charge Process
    Args:
        t_p: Pulse time (s)
        cur: Pulse charge current (A)
        ratio: Pulse ratio 
        cycle: Pulse cycle (s)
        bdy: Pulse charge SoC boundaries list
        flag_timeout: Timeout flag (0: no timeout, 1: timeout)
        data_log: Battery data log (dict)
    Return:
        data_log: Battery data log (dict)
        t_pulse: Charging time (s)
        flag_timeout: Timeout flag (0: no timeout, 1: timeout)
    '''

    t_start = data_log['t'][-1]
    soc = data_log['soc'][-1]
    volt = data_log['volt'][-1]
    egy = data_log['egy'][-1]
    volt_tau1 = data_log['volt_tau1'][-1]
    temp = data_log['temp'][-1]
    cap = data_log['cap'][-1]
        
    t = t_start

    while soc >= bdy[0] and soc <=bdy[1]:
        t_cycle = t
        while t < t_cycle + cycle * ratio:
            [soc, volt, pwr, volt_tau1, temp, cap, cap_loss, soh] = battery_model(t_p, cur, soc, volt_tau1, temp, cap)
            egy = egy + pwr * t_p / 3600

            data_log['t'].append(t)
            data_log['soc'].append(soc)
            data_log['volt'].append(volt)
            data_log['cur'].append(cur)
            data_log['pwr'].append(pwr)
            data_log['egy'].append(egy)
            data_log['volt_tau1'].append(volt_tau1)
            data_log['temp'].append(temp)
            data_log['cap'].append(cap)
            data_log['cap_loss'].append(cap_loss)
            data_log['soh'].append(soh)
            t += t_p

        while volt > 4.05:
            [soc, volt, pwr, volt_tau1, temp, cap, cap_loss, soh] = battery_model(t_p, 0, soc, volt_tau1, temp, cap)
            egy = 0

            if data_log['volt'][-1] - volt <= 3e-6:
                break

            data_log['t'].append(t)
            data_log['soc'].append(soc)
            data_log['volt'].append(volt)
            data_log['cur'].append(cur)
            data_log['pwr'].append(pwr)
            data_log['egy'].append(egy)
            data_log['volt_tau1'].append(volt_tau1)
            data_log['temp'].append(temp)
            data_log['cap'].append(cap)
            data_log['cap_loss'].append(cap_loss)
            data_log['soh'].append(soh)
            t += t_p

        if t >= 12 * 3600: # Timeout    
            flag_timeout = 1 
            break
        
    t_charge = data_log['t'][-1] - t_start

    return [t_charge, flag_timeout, data_log]

def battery_cccv_charged(cc:float, cv:float, bdy:list):
    '''Battery Constant Current Charge Process
    Args:
        cc: Constant current charge current (A)
        cv: Constant voltage charge current (V)
        bdy: Charge SoC boundaries list
    Return:
        data_log: Battery data log (dict)
    Description:
        This function is used to simulate the battery cccv charging process.
    '''

    t_p = 0.1
    data_log = {
        't': [0], 
        'soc': [0.01], 
        'volt': [3.6], 
        'cur': [cc], 
        'pwr': [0],
        'egy': [0],
        'volt_tau1': [0], 
        'temp': [298.15], 
        'cap': [3.299],
        'cap_loss': [0.001],
        'soh': [1]}

    cc_bdy = bdy[:2]
    cv_bdy = bdy[1:]
    
    [t_cc, _, data_log] = cc_charge(t_p, cc, cc_bdy, 0, data_log)
    [t_cv, _, data_log] = cv_charge(t_p, cv, cv_bdy, 0, data_log)

    return data_log

def battery_opt_charged(args:list):
    '''Battery Optimized Charge Process
    Args:
        args: Optimized charge arguments list
            [[CC1,CC2,CC3,CC4,range(0-0.2)], [iter, thread]
    Returns:
        t_cost: Charging time cost (s) list
        flag_timeout: Timeout flag (0: no timeout, 1: timeout)
        data_log: Battery data log (dict)
    Description:
        This function is used to simulate the battery optimized charging process.
    '''

    t_p = 0.1
    ratio = 0.2
    cycle = 10
    flag_timeout = 0
    data_log = {
        't': [0], 
        'soc': [0.01], 
        'volt': [3.6], 
        'cur': [0], 
        'pwr': [0],
        'egy': [0],
        'volt_tau1': [0], 
        'temp': [298.15], 
        'cap': [3.299],
        'cap_loss': [0.001],
        'soh': [1]}

    cur_cc1 = args[0][0]
    cur_cc2 = args[0][1]
    cur_cc3 = args[0][2]
    cur_pulse = args[0][3]
    range_pulse = args[0][4]

    iter = int(args[1][0])
    thread = int(args[1][1])

    dis = (1 - range_pulse)/3
    cc1_bdy = [0, dis]
    cc2_bdy = [dis, dis*2]
    cc3_bdy = [dis*2, dis*3]
    pulse_bdy = [dis*2, 0.999]

    [t_cc1, flag_timeout, data_log] = cc_charge(t_p, cur_cc1, cc1_bdy, flag_timeout, data_log)
    [t_cc2, flag_timeout, data_log] = cc_charge(t_p, cur_cc2, cc2_bdy, flag_timeout, data_log)
    [t_cc3, flag_timeout, data_log] = cc_charge(t_p, cur_cc3, cc3_bdy, flag_timeout, data_log)
    [t_pulse, flag_timeout, data_log] = pulse_charge(t_p, cur_pulse, ratio, cycle, pulse_bdy, flag_timeout, data_log)

    policy_time = [data_log['t'][-1], t_cc1, t_cc2, t_cc3, t_pulse]

    # policy_display =[round(cur_cc1,3), round(cur_cc2,3), round(cur_cc3,3), round(cur_pulse,3), round(range_pulse,3)]
    # time_display = [round(policy_time[0]/3600 ,2), round(policy_time[1]/3600,2), round(policy_time[2]/3600,2), 
    #     round(policy_time[3]/3600,2), round(policy_time[4]/3600,2)]
    # print("Iter-Num:", iter, "-", thread, "\tSoH:",  round(100 * data_log['soh'][-1], 3), "\tTemp:", 
    #     round(data_log['temp'][-1], 3), "\tPly:", policy_display,"\tTime:", time_display)
    
    return [policy_time, flag_timeout, data_log]

if __name__ == '__main__':
    
    args1 = [[1.4101585, 2.38085789, 3.3, 1.14099779, 0.2], [1,1]]
    [policy_time, data_log, flag_timeout] = battery_opt_charged(args1)