'''
Author: CLOUDUH
Date: 2022-07-16 14:20:18
LastEditors: CLOUDUH
LastEditTime: 2022-07-16 14:39:31
Description: 
'''

import matplotlib.pyplot as plt
import numpy as np

from model_photovoltaic import clac_remain_current
from process_charge import battery_opt_charged
from process_charge import battery_cccv_charged
from func.sciplt import sciplt
    
def object_function(t_cost:list, SoH:float, beta:float):
    t_m = 12*3600 # average dayttime (s)
    t_total = t_cost[0]

    J_anxiety = (t_total / t_m) * (
                0.4 * np.exp(- t_cost[1]/t_total) + 
                0.3 * np.exp(- t_cost[2]/t_total) + 
                0.2 * np.exp(- t_cost[3]/t_total) + 
                0.1 * np.exp(- t_cost[4]/t_total))

    J_SoH = (1 - SoH)

    J = beta * J_anxiety + (1 - beta) * J_SoH

    return [J, J_anxiety, J_SoH]

if __name__ == '__main__':

    # args1 = [[0.54600818, 1.10025945, 1.54652531, 6.6, 0.1], [1,1]]
    # args2 = [[0.55958162, 1.11266183, 1.66596931, 6.39172209, 0.07641518], [1,1]]
    # [t1_log, t1_cost, volt1_log, cur1_log, soc1_log, temp1_log, cap1_log, soh1_log, _] = battery_opt_charged(args1)
    # [t2_log, t2_cost, volt2_log, cur2_log, soc2_log, temp2_log, cap2_log, soh2_log, _] = battery_opt_charged(args2)
    # [t3_log, volt3_log, cur3_log, soc3_log, temp3_log, cap3_log, soh3_log] = battery_cccv_charged(1.65, 4.05, [0,0.8,1])

    args1 = [[1.4101585, 2.38085789, 3.3, 1.14099779, 0.2], [1,1]]
    args2 = [[0.987626817, 1.91887326, 2.19157007, 3.3, 0.07], [1,1]]
    args3 = [[1.65, 1.65, 1.65, 3.3, 0.0], [1,1]]

    [t1_log, t1_cost, volt1_log, cur1_log, soc1_log, temp1_log, cap1_log, soh1_log, _] = battery_opt_charged(args1)
    [t2_log, t2_cost, volt2_log, cur2_log, soc2_log, temp2_log, cap2_log, soh2_log, _] = battery_opt_charged(args2)
    [t3_log, t3_cost, volt3_log, cur3_log, soc3_log, temp3_log, cap3_log, soh3_log, _] = battery_opt_charged(args3)
    # [t3_log, volt3_log, cur3_log, soc3_log, temp3_log, cap3_log, soh3_log] = battery_cccv_charged(1.65, 4.05, [0,0.8,1])

    [t_pv, cur_pv, pwr_pv, egy_pv, t_riseup, t_falldown] = clac_remain_current(0.1, 180, 30)

    J_1 = object_function(t1_cost, soh1_log[-1], 0.5)
    J_2 = object_function(t2_cost, soh2_log[-1], 0.5)
    J_3 = object_function(t3_cost, soh3_log[-1], 0.5)
    print("except:", J_1, "\nnone:", J_2, "\nbitch:", J_3)

    x_limit = [0, max(t1_log[-1], t2_log[-1])+200]

    plt.subplot(231)
    volt_plot = [[t1_log,volt1_log,'PSO-Fixed','o','r',5000,0.01], [t2_log,volt2_log,'PSO-Variable','v','g',5000,0.01], [t3_log,volt3_log,'CCCV','s','b',5000,0.01]]
    sciplt(volt_plot, "Time(s)", "Voltage(V)", "Terminal Voltage", "lower right", x_limit, [2.8,4.3])

    plt.subplot(232)
    cur_plot = [[t1_log,cur1_log,'PSO-Fixed','o','r',5000,0.01], [t2_log,cur2_log,'PSO-Variable','v','g',5000,0.01], [t3_log,cur3_log,'CCCV','s','b',5000,0.01],[t_pv, cur_pv,'PV','^','k',5000,0.01]]
    sciplt(cur_plot, "Time(s)", "Current(A)", "Current", "upper left", x_limit, [0,3.5])

    plt.subplot(233)
    soc_plot = [[t1_log,soc1_log,'PSO-Fixed','o','r',5000,0.01], [t2_log,soc2_log,'PSO-Variable','v','g',5000,0.01], [t3_log,soc3_log,'CCCV','s','b',5000,0.01]]
    sciplt(soc_plot, "Time(s)", "SOC(%)", "State of Charge", "lower right", x_limit, [0,1])

    plt.subplot(234)
    temp_plot = [[t1_log,temp1_log,'PSO-Fixed','o','r',5000,0.01], [t2_log,temp2_log,'PSO-Variable','v','g',5000,0.01], [t3_log,temp3_log,'CCCV','s','b',5000,0.01]]
    sciplt(temp_plot, "Time(s)", "Temperature(K)", "Temperature", "lower right", x_limit, [280,320])

    plt.subplot(235)
    cap_plot = [[t1_log,cap1_log,'PSO-Fixed','o','r',5000,0.01], [t2_log,cap2_log,'PSO-Variable','v','g',5000,0.01], [t3_log,cap3_log,'CCCV','s','b',5000,0.01]]
    sciplt(cap_plot, "Time(s)", "Capacity(Ah)", "Capacity", "upper right", x_limit, [3.0,3.4])

    plt.subplot(236)
    soh_plot = [[t1_log,soh1_log,'PSO-Fixed','o','r',5000,0.01], [t2_log,soh2_log,'PSO-Variable','v','g',5000,0.01], [t3_log,soh3_log,'CCCV','s','b',5000,0.01]]
    sciplt(soh_plot, "Time(s)", "SOH(%)", "State of Health", "upper right", x_limit, [0.6,1])
    plt.show()