'''
Author: CLOUDUH
Date: 2022-07-09 14:58:26
LastEditors: CLOUDUH
LastEditTime: 2022-07-16 15:03:04
Description: 
'''

import matplotlib.pyplot as plt
import numpy as np

from model_battery import battery_model

def cc_charge(t_p:float, cur:float, bdy:list, flag_timeout:int,
    t_log:list, volt_log:list, cur_log:list, soc_log:list, volt_tau1_log:list, temp_log:list, cap_log:list, soh_log:list):
    '''Battery Constant Current Charge Process
    Args:
        t_p: Pulse time (s)
        cur: Pulse charge current (A)
        bdy: Pulse charge SoC boundaries list
        t_log: Charging time cost (s) list
        volt_log: Battery terminal voltage (V) list
        soc_log: Battery state of charge list
        cap_log: Battery capacity (Ah) list
        soh_log: Battery State of Health list
        temp_log: Battery temperature (K) list
    Return:
        t_log: Charging time cost (s) list
        volt_log: Battery terminal voltage (V) list
        soc_log: Battery state of charge list
        cap_log: Battery capacity (Ah) list
        soh_log: Battery State of Health list
        temp_log: Battery temperature (K) list
        t_charge: Charging time (s)
        flag_timeout: Timeout flag (0: no timeout, 1: timeout)
    '''
 
    soc = soc_log[-1]
    temp = temp_log[-1]
    cap = cap_log[-1]
    t_start = t_log[-1]
    volt_tau1 = volt_tau1_log[-1]

    t = t_start

    while soc >= bdy[0] and soc <=bdy[1]:
        [volt, soc, volt_tau1, temp, cap, cap_loss, soh] = battery_model(t_p, cur, soc, volt_tau1, temp, cap)
        t_log.append(t)
        volt_log.append(volt)
        cur_log.append(cur)
        soc_log.append(soc)
        volt_tau1_log.append(volt_tau1)
        temp_log.append(temp)
        cap_log.append(cap)
        soh_log.append(soh)
        t = t + t_p

        if t >= 12 * 3600: # Timeout    
            flag_timeout = 1 
            break
    
    t_charge = t_log[-1] - t_start

    return [t_log, volt_log, cur_log, soc_log, volt_tau1_log, temp_log, cap_log, soh_log, t_charge, flag_timeout]

def cv_charge(t_p:float, cv:float, bdy:list, flag_timeout:int,
    t_log:list, volt_log:list, cur_log:list, soc_log:list, volt_tau1_log:list, temp_log:list, cap_log:list, soh_log:list):
    '''Battery Constant Voltage Charge Process
    Args:
        t_p: Pulse time (s)
        cur: Pulse charge current (A)
        bdy: Pulse charge SoC boundaries list
        volt_log: Battery terminal voltage (V) list
        soc_log: Battery state of charge list
        t_loss_log: Charging time cost (s) list
        cap_log: Battery capacity (Ah) list
        soh_log: Battery State of Health list
        temp_log: Battery temperature (K) list
    Return:
        t_log: Charging time cost (s) list
        volt_log: Battery terminal voltage (V) list
        soc_log: Battery state of charge list
        cap_log: Battery capacity (Ah) list
        soh_log: Battery State of Health list
        temp_log: Battery temperature (K) list
        t_charge: Charging time (s)
        flag_timeout: Timeout flag (0: no timeout, 1: timeout)
    '''

    soc = soc_log[-1]
    temp = temp_log[-1]
    cap = cap_log[-1]
    t_start = t_log[-1]
    volt = volt_log[-1]
    volt_tau1 = volt_tau1_log[-1]

    t = t_start
    cur = 1.65

    while soc > bdy[0] and soc <=bdy[1]:

        if volt > cv:
            cur = cur - 1e-2
        else:
            pass

        if cur <= 6.5e-2: break

        [volt, soc, volt_tau1, temp, cap, cap_loss, soh] = battery_model(t_p, cur, soc, volt_tau1, temp, cap)
        t_log.append(t)
        volt_log.append(volt)
        cur_log.append(cur)
        soc_log.append(soc)
        volt_tau1_log.append(volt_tau1)
        temp_log.append(temp)
        cap_log.append(cap)
        soh_log.append(soh)
        t = t + t_p

        if t >= 12 * 3600: # Timeout    
            flag_timeout = 1 
            break
    
    t_charge = t_log[-1] - t_start

    return [t_log, volt_log, cur_log, soc_log, volt_tau1_log, temp_log, cap_log, soh_log, t_charge, flag_timeout]

def pulse_charge(t_p:float, cur:float, ratio:float, cycle:float, bdy:list, flag_timeout:int,
    t_log:list, volt_log:list, cur_log:list, soc_log:list, volt_tau1_log:list, temp_log:list, cap_log:list, soh_log:list):
    '''Battery Pulse Charge Process
    Args:
        t_p: Pulse time (s)
        cur: Pulse charge current (A)
        ratio: Pulse ratio 
        cycle: Pulse cycle (s)
        bdy: Pulse charge SoC boundaries list
        volt_log: Battery terminal voltage (V) list
        soc_log: Battery state of charge list
        t_loss_log: Charging time cost (s) list
        cap_log: Battery capacity (Ah) list
        soh_log: Battery State of Health list
        temp_log: Battery temperature (K) list
    Return:
        t_log: Charging time cost (s) list
        volt_log: Battery terminal voltage (V) list
        soc_log: Battery state of charge list
        cap_log: Battery capacity (Ah) list
        soh_log: Battery State of Health list
        temp_log: Battery temperature (K) list
        t_pulse: Charging time (s)
        flag_timeout: Timeout flag (0: no timeout, 1: timeout)
    '''

    soc = soc_log[-1]
    temp = temp_log[-1]
    cap = cap_log[-1]
    t_start = t_log[-1]
    volt_tau1 = volt_tau1_log[-1]
        
    t = t_start

    while soc >= bdy[0] and soc <=bdy[1]:
        t_cycle = t
        while t < t_cycle + cycle * ratio:
            [volt, soc, volt_tau1, temp, cap, cap_loss, soh] = battery_model(t_p, cur, soc, volt_tau1, temp, cap)
            t_log.append(t)
            volt_log.append(volt)
            cur_log.append(cur)
            soc_log.append(soc)
            volt_tau1_log.append(volt_tau1)
            temp_log.append(temp)
            cap_log.append(cap)
            soh_log.append(soh)
            t += t_p

        while volt > 4.05:
            [volt, soc, volt_tau1, temp, cap, cap_loss, soh] = battery_model(t_p, 0, soc, volt_tau1, temp, cap)

            if volt_log[-1] - volt <= 3e-6:
                break

            t_log.append(t)
            volt_log.append(volt)
            cur_log.append(0)
            soc_log.append(soc)
            volt_tau1_log.append(volt_tau1)
            temp_log.append(temp)
            cap_log.append(cap)
            soh_log.append(soh)
            t += t_p

        if t >= 12 * 3600: # Timeout    
            flag_timeout = 1 
            break
        
    t_charge = t_log[-1] - t_start

    return [t_log, volt_log, cur_log, soc_log, volt_tau1_log, temp_log, cap_log, soh_log, t_charge, flag_timeout]

def battery_cccv_charged(cc:float, cv:float, bdy:list):
    '''Battery Constant Current Charge Process
    Args:
        cc: Constant current charge current (A)
        cv: Constant voltage charge current (V)
        bdy: Charge SoC boundaries list
    Return:
        volt_log: Battery terminal voltage (V) list
        soc_log: Battery state of charge list
        t_loss_log: Charging time cost (s) list
        cap_log: Battery capacity (Ah) list
        soh_log: Battery State of Health list
        temp_log: Battery temperature (K) list
    '''

    t_p = 0.1
    volt_log = [3.6]
    cur_log = [cc]
    soc_log = [0.01]
    volt_tau1_log = [0]
    temp_log = [298.15]
    cap_log = [3.299]
    t_log = [0]
    soh_log = [1]

    cc_bdy = bdy[:2]
    [t_log, volt_log, cur_log, soc_log, volt_tau1_log, temp_log, cap_log, soh_log, t_cc, _] = \
        cc_charge(t_p, cc, cc_bdy, 0, t_log, volt_log, cur_log, soc_log, volt_tau1_log, temp_log, cap_log, soh_log)

    cv_bdy = bdy[1:]
    [t_log, volt_log, cur_log, soc_log, volt_tau1_log, temp_log, cap_log, soh_log, t_cv, _] = \
        cv_charge(t_p, cv, cv_bdy, 0, t_log, volt_log, cur_log, soc_log, volt_tau1_log, temp_log, cap_log, soh_log)

    return [t_log, volt_log, cur_log, soc_log, temp_log, cap_log, soh_log]

def battery_opt_charged(args:list):
    '''Battery Optimized Charge Process
    Args:
        args: Optimized charge arguments list
            [[CC1,CC2,CC3,CC4,range(0-0.2)], [iter, thread]
    Returns:
        t_log: Charging time cost (s) list
        volt_log: Battery terminal voltage (V) list
        soc_log: Battery state of charge list
        cap_log: Battery capacity (Ah) list
        soh_log: Battery State of Health list
        temp_log: Battery temperature (K) list
        t_pulse: Charging time (s)
        flag_timeout: Timeout flag (0: no timeout, 1: timeout)
    Description:
        Optimized charge process
    '''

    t_p = 0.1
    volt_log = [3.6]
    cur_log = [args[0][0]]
    soc_log = [0.01]
    volt_tau1_log = [0]
    temp_log = [298.15]
    cap_log = [3.299]
    t_log = [0]
    soh_log = [1]
    ratio = 0.2
    cycle = 10

    cur_cc1 = args[0][0]
    cur_cc2 = args[0][1]
    cur_cc3 = args[0][2]
    cur_pulse = args[0][3]

    iter = int(args[1][0])
    thread = int(args[1][1])

    dis = (1 - range_pulse)/3

    cc1_bdy = [0, dis]
    cc2_bdy = [dis, dis*2]
    cc3_bdy = [dis*2, dis*3]
    pulse_bdy = [dis*2, 0.999]

    [t_log, volt_log, cur_log, soc_log, volt_tau1_log, temp_log, cap_log, soh_log, t_cc1, flag_timeout] = \
        cc_charge(t_p, cur_cc1, cc1_bdy, 0, t_log, volt_log, cur_log, soc_log, volt_tau1_log, temp_log, cap_log, soh_log)

    [t_log, volt_log, cur_log, soc_log, volt_tau1_log, temp_log, cap_log, soh_log, t_cc2, flag_timeout] = \
        cc_charge(t_p, cur_cc2, cc2_bdy, flag_timeout, t_log, volt_log, cur_log, soc_log, volt_tau1_log, temp_log, cap_log, soh_log)

    [t_log, volt_log, cur_log, soc_log, volt_tau1_log, temp_log, cap_log, soh_log, t_cc3, flag_timeout] = \
        cc_charge(t_p, cur_cc3, cc3_bdy, flag_timeout, t_log, volt_log, cur_log, soc_log, volt_tau1_log, temp_log, cap_log, soh_log)

    [t_log, volt_log, cur_log, soc_log, volt_tau1_log, temp_log, cap_log, soh_log, t_pulse, flag_timeout] = \
        pulse_charge(t_p, cur_pulse, ratio, cycle, pulse_bdy, flag_timeout, t_log, volt_log, cur_log, soc_log, volt_tau1_log, temp_log, cap_log, soh_log)

    t_cost = [t_log[-1], t_cc1, t_cc2, t_cc3, t_pulse]

    policy_display =[round(cur_cc1,3), round(cur_cc2,3), round(cur_cc3,3), round(cur_pulse,3), round(range_pulse,3)]
    t_cost_display = [round(t_cost[0]/3600 ,2), round(t_cost[1]/3600,2), round(t_cost[2]/3600,2), round(t_cost[3]/3600,2), round(t_cost[4]/3600,2)]
    
    # print("Thread:", thread, "Policy:", policy_display, "Time Cost:", t_cost_display)
    # print("Iter-Num:", iter, "-", thread, "\tSoH:",  round(100 * soh_log[-1], 3), "\tTemp:", 
    #     round(temp_log[-1], 3), "\tPly:", policy_display,"\tTime:", t_cost_display)
    
    return [t_log, t_cost, volt_log, cur_log, soc_log, temp_log, cap_log, soh_log, flag_timeout]

if __name__ == '__main__':
    
    [t1_log, t1_cost, volt1_log, cur1_log, soc1_log, temp1_log, cap1_log, soh1_log, _] = battery_opt_charged([1.65, 1.65, 1.65, 3.3, 0.0])