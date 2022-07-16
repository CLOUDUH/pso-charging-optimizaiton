'''
Author: CLOUDUH
Date: 2022-07-11 17:08:37
LastEditors: CLOUDUH
LastEditTime: 2022-07-11 17:42:49
Description: 
'''
from matplotlib import gridspec
from algorithm_pso import particle_swarm_optimization
from func.sciplt import sciplt
import matplotlib.pyplot as plt

if __name__ == '__main__':
    N = 20
    d = 5
    ger = 50
    beta_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    J1_opt = []
    J2_opt = []
    J3_opt = []
    policy1_opt = []
    policy2_opt = []
    policy3_opt = []
    policy4_opt = []
    policy5_opt = []
    t1_opt = []
    t2_opt = []
    t3_opt = []
    t4_opt = []
    t5_opt = []

    for beta in beta_list:
        [policy_log, policy_seek, policy_swarm, J_log, J_seek, J_swarm, t_log, t_seek, t_swarm] = \
            particle_swarm_optimization(N, d, ger, beta)
        J1_opt.append(J_swarm[0])
        J2_opt.append(J_swarm[1])
        J3_opt.append(J_swarm[2])
        policy1_opt.append(policy_swarm[0])
        policy2_opt.append(policy_swarm[1])
        policy3_opt.append(policy_swarm[2])
        policy4_opt.append(policy_swarm[3])
        policy5_opt.append(policy_swarm[4])
        t1_opt.append(t_swarm[0])
        t2_opt.append(t_swarm[1])
        t3_opt.append(t_swarm[2])
        t4_opt.append(t_swarm[3])
        t5_opt.append(t_swarm[4])

    G = gridspec.GridSpec(2, 2)
    plt.subplot(G[0,:])
    J_plot = [[beta_list,J1_opt,'JTotal','o','r',1,0.01], [beta_list,J2_opt,'JTime','^','g',1,0.01], [beta_list,J2_opt,'JSoH','s','b',1,0.01]]
    sciplt(J_plot, "Beta", "Value", "Cost Function", "lower right", [0,1.1], [0,1])

    plt.subplot(G[1,0])
    policy_plot = [[beta_list,policy1_opt,'CC1','o','r',1,0.01], [beta_list,policy2_opt,'CC2','^','g',1,0.01], [beta_list,policy3_opt,'CC3','s','b',1,0.01]
                    , [beta_list,policy4_opt,'Pulse','o','r',1,0.01], [beta_list,policy5_opt,'Range','^','g',1,0.01]]
    sciplt(policy_plot, "Beta", "Value", "Policy", "lower right", [0,1.1], [0,6.6])

    plt.subplot(G[1,1])
    t_plot = [[beta_list,t1_opt,'TimeTotal','o','r',1,0.01], [beta_list,t2_opt,'CC1','^','g',1,0.01], [beta_list,t3_opt,'CC2','s','b',1,0.01]
                , [beta_list,t4_opt,'CC3','o','r',1,0.01], [beta_list,t5_opt,'Pulse','^','g',1,0.01]]
    sciplt(t_plot, "Beta", "Value", "Time", "lower right", [0,1.1], [0,10000])
    
    plt.show()