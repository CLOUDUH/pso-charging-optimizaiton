'''
Author: CLOUDUH
Date: 2022-05-28 17:55:32
LastEditors: CLOUDUH
LastEditTime: 2022-07-16 16:25:43
Description: Battery charging optimization by PSO
    Battery charging optimization program.
    Optimization algorithm is particle swarm optimization
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import multiprocessing as mp

from func.sciplt import sciplt
from model_photovoltaic import clac_remain_current
from process_charge import battery_opt_charged

Qe = 3.3 # Battery Capacity (Ah)
ratio_pulse = 0.2 # Duty ratio of pulse charging

def match_policy(policy:list, t_pv_list:list, cur_pv_list:list):
    '''Calculate Remain Current
    Args:
        policy: Policy list
        t_pv_list: Time list
        cur_pv_list: Current list
    Returns:   
        t_remain_list: Remain time list (s)
        cur_remain_list: Remain current list (A)
    '''

    try:
        t1 = np.interp(policy[0], cur_pv_list, t_pv_list)
    except:
        t1 = np.inf
 
    t2 = t1 + 0.2 * 3600 * Qe / policy[0]
    t3 = t2 + 0.2 * 3600 * Qe / policy[1]
    t4 = t3 + 0.2 * 3600 * Qe / policy[2]
    t5 = t4 + 0.2 * 3600 * Qe / (policy[3] * ratio_pulse)

    t_list = [t1, t2, t3, t4, t5] # The time point of each charge 

    cur_max_1 = max(np.interp(t1, t_pv_list, cur_pv_list), np.interp(t2, t_pv_list, cur_pv_list))
    cur_max_2 = max(np.interp(t2, t_pv_list, cur_pv_list), np.interp(t3, t_pv_list, cur_pv_list))
    cur_max_3 = max(np.interp(t3, t_pv_list, cur_pv_list), np.interp(t4, t_pv_list, cur_pv_list))
    cur_max_4 = max(np.interp(t4, t_pv_list, cur_pv_list), np.interp(t5, t_pv_list, cur_pv_list))

    cur_list = [cur_max_1, cur_max_2, cur_max_3, cur_max_4]

    return [t_list, cur_list]

def obj_func(SoH:float, t_cost:list, flag:int, policy:list, beta:float, t_remain_list:list, cur_remain_list:list):
    '''Object Function Calculate
    Args: 
        SoH: State of Health
        t: Total charging time list (s)
        flag: Timeout
        policy: Charging policy (list)
        beta: Weight coeffeicent 0-1
        cur_remain_list: Remain current list (A)
    Returns:
        J: Cost
        J_anxiety: Anxiety cost
        J_SoH: SoH cost
    Description:
        The transformation of the objective function is 
        determined by the weight coefficients.
        except before that, when the particle 
        exceeds the limit, it will become infinite
    '''
    t_m = 12*3600 # average dayttime (s)
    t_total = t_cost[0]

    J_anxiety = (t_total / t_m) * (
                0.4 * np.exp(- t_cost[1]/t_total) + 
                0.3 * np.exp(- t_cost[2]/t_total) + 
                0.2 * np.exp(- t_cost[3]/t_total) + 
                0.1 * np.exp(- t_cost[4]/t_total))
                
    J_SoH = (1 - SoH)

    J = beta * J_anxiety + (1 - beta) * J_SoH

    # print(J_anxiety, J)

    if flag == 1: J = np.inf
    if t_total > 10 * 3600: J = np.inf
    if SoH < 0: J = np.inf
    if t_remain_list[0] == np.inf: J = np.inf
    
    for i in policy:
        if i == 0: J = np.inf

    for i in range(len(cur_remain_list)):
        if t_remain_list[i] < policy[i]:
            print("111111111") #TODO
            J = np.inf
            break

    return [J, J_anxiety, J_SoH]

def particle_swarm_optimization(N:int, d:int, ger:int, beta:float):
    '''Particle swarm optimization
    Args:
        N: Particle swarm number
        d: Particle dimension
        ger: Number of iterations
    Returns:
        policy_log: Charging policy (ger-N-d)
        policy_particle: Charging policy (N-d)
        policy_swarm: Charging policy (1-d)
        J_log: Objective function value of each iteration of each particle (ger-N)
        J_particle: Optimal objective function value of particle (N-1)
        J_swarm: Global optimal objective function value (1)
        t_log: charging time list (ger-N-d+1)
        t_particle: charging time list (N-d+1)
        t_swarm: Global optimal charging time (1-d+1)
    Description:
        The particle swarm optimization is a method of optimization.
    '''

    date = 180 # Today 0-365
    latitude = 30 # (°)
    vol_bat = 3.6 # Battery voltage (V)
    w = 0.729 # Inertial weight
    c1 = 1.49115 # Self learning factor
    c2 = 1.49115 # Swarm learning factor 
    # beta = 0.5 # Weight coefficient 1: fastest; 0: healthiest
    iter = 0  # Initial iteration
    np.random.seed(N * d * ger) # set seed

    # Initialize the particle position and velocity
    x = np.zeros((N,d)) # Particle Position (N-d)
    v = np.zeros((N,d)) # Particle Velcocity (N-d)

    [t_pv, cur_pv, pwr_pv, egy_pv, t_riseup, t_falldown] = clac_remain_current(0.1, 180, 30) #

    policy_limit_cc = np.matlib.repmat(np.array([[0],[3.3]]),1,d-2) 
    policy_limit_pulse = np.array([[0],[3.3]])
    policy_limit_range = np.array([[0],[0.2]])
    policy_limit = np.hstack((policy_limit_cc, policy_limit_pulse, policy_limit_range)) # Charging crrent limits (2-d)

    vlimit_cc = np.matlib.repmat(np.array([[-0.33],[0.33]]),1,d-2) 
    vlimit_pulse = np.array([[-0.5],[0.5]])
    vlimit_range = np.array([[-0.01],[0.01]])
    vlimit = np.hstack((vlimit_cc, vlimit_pulse, vlimit_range)) # Velocity limits (2-d)

    # Initialize the particle position
    for i in range(d):
        x[:,i] = np.matlib.repmat(policy_limit[0,i],1,N) + \
                (policy_limit[1,i] - policy_limit[0,i]) * np.random.rand(1,N)  
    # Initialize the particle velocity
    for i in range(d):
        v[:,i] = np.matlib.repmat(vlimit[0,i],1,N) + \
                (vlimit[1,i] - vlimit[0,i]) * np.random.rand(1,N)
        
    policy_log = np.zeros((ger,N,d)) # Policy of each particles-iteration (ger-N-d)
    policy_particle = x # Best policy of particle (N-d)
    policy_swarm = np.zeros((1,d)) # Best policy of swarm (1-d)
    policy_seek = np.zeros((ger,d)) # save each iteration best policy (ger)
    
    J_log = np.zeros((ger,N,3)) # Objective function value of particles-iteration (ger-N-3)
    J_particle = np.inf * np.ones((N,3)) # Optimal objective function value of particle (N-1)
    J_swarm = [1, 0, 0] # Optimal objective function value of swarm (1)
    J_seek = np.zeros((ger,3)) # save each iteration best objective function value (ger)

    t_log = np.zeros((ger,N,d)) # Charging time of each particles-iteration (ger-N-d)
    t_particle = np.zeros((N,d)) # Charging time of particle (N)
    t_swarm = np.zeros((1,d)) # Charging time of swarm (1)
    t_seek = np.zeros((ger,d)) # save each iteration best charging time (ger)

    while iter <= ger - 1:

        pool = mp.Pool() # create a multiprocessing pool
        processes = [] # create a list of processes

        for j in range(N): 
            args = [x[j].tolist(), [iter,j]]
            processes.append(args)

        results = pool.map(battery_opt_charged, processes) # map the function to the pool !!!

        pool.close() # Parallel computing
        pool.join() # Wait all thread to finish

        for j in range(N): 
            [t_log[iter,j], flag, data_log] = results[j]
            SoH = data_log['soh'][-1]

            [t_remian, cur_remain] = match_policy(x[j], t_pv, cur_pv) # match the policy
            
            policy_log[iter,j] = x[j] # save the policy
            
            J_log[iter,j] = obj_func(SoH, t_log[iter,j], flag, x[j], beta, t_remian, cur_remain) # Optimal function value

            if J_particle[j,0] > J_log[iter,j,0]:
                J_particle[j] = J_log[iter,j]
                policy_particle[j] = x[j]
                t_particle[j] = t_log[iter,j]
            
        if J_swarm[0] > np.amin(J_particle[:,0]):
            nmin = np.argmin(J_particle[:,0])
            J_swarm = J_particle[nmin]
            policy_swarm = policy_particle[nmin]
            t_swarm = t_particle[nmin]

        v = v * w + c1 * np.random.rand() * (policy_particle - x) + \
                    c2 * np.random.rand() * (np.matlib.repmat(policy_swarm,N,1) - x)

        # Position saturation
        for k in range(d):
            hsat = np.where(v[:,k] < vlimit[0,k], 1, 0) #
            lsat = np.where(v[:,k] > vlimit[1,k], 2, 0)

            for j in range(N):
                if hsat[j] == 1:
                    v[j,k] = vlimit[0,k]
                elif lsat[j] == 2:
                    v[j,k] = vlimit[1,k]

        x = x + v # Updating position

        # Velcocity saturation
        for k in range(d): 
            hsat = np.where(x[:,k] < policy_limit[0,k], 1, 0)
            lsat = np.where(x[:,k] > policy_limit[1,k], 2, 0)

            for j in range(N):
                if hsat[j] == 1:
                    x[j,k] = policy_limit[0,k]
                elif lsat[j] == 2:
                    x[j,k] = policy_limit[1,k]

        print("Iteration:",iter, "\nPolicy of Swarm:\n", policy_swarm, 
            "\nTime of Swarm:\n", t_swarm, "\nFunction Value of Swarm:\n", J_swarm, "\n")
        
        # print("Beta:", beta, "Iteration:", iter, "Policy:", policy_swarm)

        policy_seek[iter] = policy_swarm
        J_seek[iter] = J_swarm
        t_seek[iter] = t_swarm

        iter = iter + 1

    return [policy_log, policy_seek, policy_swarm, J_log, J_seek, J_swarm, t_log, t_seek, t_swarm]

if __name__ == '__main__':

    N = 20
    d = 5
    ger = 50

    [policy_log, policy_seek, policy_swarm, J_log, J_seek, J_swarm, t_log, t_seek, t_swarm] = \
        particle_swarm_optimization(N, d, ger, 0.5)

    iter = np.arange(0,ger)

    G = gridspec.GridSpec(2, 2)
    plt.subplot(G[0,:])
    iter_plot =[
        [iter, J_seek[:,0],'J Total','v','r',1,0.01],
        [iter, J_seek[:,1],'J Time','v','g',1,0.01],
        [iter, J_seek[:,2],'J SoH2','v','b',1,0.01]]
    sciplt(iter_plot, "Iteration", "J", "Objective Function Value of Particle", "lower right", [0,30], [0,1])

    plt.subplot(G[1,0])
    policy_plot = [
        [iter, policy_seek[:,0],'CC1','v','r',1,0.01], 
        [iter, policy_seek[:,1],'CC2','v','g',1,0.01],
        [iter, policy_seek[:,2],'CC3','v','b',1,0.01],
        [iter, policy_seek[:,3],'Pulse','v','y',1,0.01]]
    sciplt(policy_plot, "Iteration", "Policy(A)", "Policy of Particle", "lower right", [0,30], [0,6])

    plt.subplot(G[1,1])
    t_plot = [
        [iter, t_seek[:,0],'Total','v','w',1,0.01],
        [iter, t_seek[:,1],'CC1','v','r',1,0.01],
        [iter, t_seek[:,2],'CC2','v','g',1,0.01],
        [iter, t_seek[:,3],'CC3','v','b',1,0.01],
        [iter, t_seek[:,4],'Pulse','v','y',1,0.01]]
    sciplt(t_plot, "Iteration", "Time(t)", "Charging Time of Particle", "lower right", [0,30], [0,5000])

    plt.show()