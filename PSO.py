import sys
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import numpy.matlib


## Initialization

# Load battert basic value
VO_Tab = pd.read_csv("utils/VO.csv",header=None)
R0_Tab = pd.read_csv("utils/R0.csv",header=None)
R1_Tab = pd.read_csv("utils/R1.csv",header=None)
tau1_Tab = pd.read_csv("utils/tau1.csv",header=None)
Grid = pd.read_csv("utils/Grid.csv",header=None)

t_p = 1  # Step
Tf = 298.15 # Homoeothermy
Qe = 3.3 # Battery Capacity Ah
t_m = 2*3600 # Empirical charging time
h = 40.106 # Heat transfer coefficient W/(m^2*K)
c = 800 # Specific heat capacity J/(kg*K)
A = 0.004317 # Heat exchange area at the cell surface m^2
m =0.0475 # Battery mass kg
z = 0.4 # Order of Ah throughput
B = 130 # Pre-exponential factor
E_a = 18461 # Activation energy for cycle aging J/mol
R = 8.314 # Ideal gas constant J/(kg*K)
alpha = 32 # Coefficient for aging acceleration caused by the current

# def PSO(N, d, ger):

#     x = np.zeros((N,d))
#     v = np.zeros((N,d))
#     iter = 1
#     w = 0.729
#     c1 = 1.49115
#     c2 = 1.49115
#     beta = 1

#     Ilimit = np.matlib.repmat(np.array([[0],[3.3]]),1,d)
#     vlimit = np.matlib.repmat(np.array([[- 0.33],[0.33]]),1,d)

#     lsat = np.zeros((N,d))
#     hsat = np.zeros((N,d))



# # f= @(SoH, t)(1 - beta) * (1 - SoH) + beta * t / t_m; # Objective function
#     f1 = lambda t = None: beta * t / t_m

#     for i in np.arange(1,d+1).reshape(-1):
#         if i == 1:
#             x[:,i] = np.matlib.repmat(Ilimit(1,i),N,1) + (Ilimit(2,i) - Ilimit(1,i)) * np.random.rand(N,1)
#         else:
#             x[:,i] = np.matlib.repmat(Ilimit(1,i),N,1) + (x(i - 1) - Ilimit(1,i)) * np.random.rand(N,1)
#         v[:,i] = np.matlib.repmat(vlimit(1,i),N,1) + (vlimit(2,i) - vlimit(1,i)) * np.random.rand(N,1)

#     xm = x
#     ym = np.zeros((1,d))
#     SoH = np.zeros((ger,N))
#     t = np.zeros((ger,N))
#     fx = np.zeros((ger,N))
#     fxm = np.zeros((N,1))
#     fym = - inf

#     while iter <= ger:

#         for j in np.arange(1,N+1).reshape(-1):
#             SoH[iter,j],t[iter,j] = BatChrg(x(j,:))
#             # fx(iter, j) = f(SoH(iter, j), t(iter,j)) # Optimal function value
#             fx[iter,j] = f1(t(iter,j))
#             if fxm(j) > fx(iter,j):
#                 fxm[j] = fx(iter,j)
#                 xm[j,:] = x(j,:)
#         if fym > np.amin(fxm):
#             fym,nmin = np.amin(fxm)
#             ym = xm(nmin,:)
#         v = v * w + c1 * rand * (xm - x) + c2 * rand * (np.matlib.repmat(ym,N,1) - x)
#         for k in np.arange(1,d+1).reshape(-1):
#             v[v[:,k] < vlimit[1,k]] = vlimit(1,k)
#             v[v[:,k] > vlimit[2,k]] = vlimit(2,k)
#         x = x + v
#         for k in np.arange(1,d+1).reshape(-1):
#             x[x[:,k] < Ilimit[1,k]] = Ilimit(1,k)
#             x[x[:,k] > Ilimit[2,k]] = Ilimit(2,k)
#         x
#         iter = iter + 1

def ECM(I = None, Temp = None, SoC = None): 
    """Battery 1-RC Equivalent Circuit Model
    Input:
        I: Battery current(Charging positive)
        Temp: Battery temperature
        SoC: State of Charge
    Output:
        Vt: Battery Voltage
        SoC: State of Charge
    """

    temp = Temp - 273.15
    VO = griddata(Grid, VO_Tab, (SoC, temp), method='linear')
    R0 = griddata(Grid, R0_Tab, (SoC, temp), method='linear')
    tau1 = griddata(Grid, tau1_Tab, (SoC, temp), method='linear')
    Vt = VO + I * (1 - np.exp(t_p / tau1)) + I * R0
    SoC = SoC + t_p * I / (3600 * Qe)
    return [Vt, SoC]

def ThM(I = None, Temp = None, SoC = None): 
    """Battery Thermal Model
    Input:
        I: Battery current(Charging positive)
        Temp: Battery temperature
        SoC: State of Charge
    Output:
        Temp: Battery temperature
    """

    temp = Temp - 273.15
    R0 = griddata(Grid, R0_Tab, (SoC, temp), method='linear')
    R1 = griddata(Grid, R1_Tab, (SoC, temp), method='linear')
    dVO_T = (13.02 * SoC ** 5 - 38.54 * SoC ** 4 + 32.81 * SoC ** 3 + 2.042 \
        * SoC ** 2 - 13.93 * SoC + 4.8) / 1000
    dTemp = (I ** 2 * (R0 + R1) + I * Temp * dVO_T - h * A * (Temp - Tf)) / (m * c)
    Temp = Temp + dTemp * t_p
    return Temp

def AM(I = None, Temp = None, Qloss = None): 
    """Battery Aging Model
    Input:
        I: Battery current(Charging positive)
        Temp: Battery temperature
        Qloss: Loss battery capacity
    Output:
        Qloss: Loss battery capacity
        SoH: State of Health
    """

    dQloss = (np.abs(I) / 3600) * z * B * np.exp((- E_a + alpha * np.abs(I)) / (R * Temp)) \
        * (Qloss / (B * np.exp((- E_a + alpha * np.abs(I)) / (R * Temp)))) ** (1 - (1 / z))
    Qloss = Qloss + dQloss * t_p
    SoH = 1 - ((Qloss / Qe) / 0.2)
    return [Qloss, SoH]

def BatChrg(CC = None): 
    """Battery Charging Function
    Input:
        CC: Battery charging constant-current(5-1 matrix)
    Output:
        SoH: Whole charging process SoH
        t: Charging time cost
    """

    SoC = 0.1
    Qloss = 0.0001
    SoH = 1 - ((Qloss / Qe) / 0.2)
    Temp = 298.15
    i = 1

    SoCRange = np.array([0.2,0.4,0.6,0.8,1.0])

    SoC,Qloss,SoH,Temp,i = nCC(CC(1),SoC,SoCRange(1),Qloss,SoH,Temp,i)
    SoC,Qloss,SoH,Temp,i = nCC(CC(2),SoC,SoCRange(2),Qloss,SoH,Temp,i)
    SoC,Qloss,SoH,Temp,i = nCC(CC(3),SoC,SoCRange(3),Qloss,SoH,Temp,i)
    SoC,Qloss,SoH,Temp,i = nCC(CC(4),SoC,SoCRange(4),Qloss,SoH,Temp,i)
    SoC,Qloss,SoH,Temp,i = nCC(CC(5),SoC,SoCRange(5),Qloss,SoH,Temp,i)
    t = (i - 1) * 0.05
    return SoH,t

def nCC(I = None,SoC = None,SoCRange = None,Qloss = None,SoH = None,Temp = None,i = None): 
    """Battery n-Constant Current Charging Process
    Input:
        I: Battery current(Charging positive)
        SoC: State of Charge
        SoCRange: n-CC Charging SoC Range
        Qloss: Loss battery capacity
        SoH: State of Health
        Temp: Battery temperature
        i: Flag of cycle
    Output:
        SoC: State of Charge
        Qloss: Loss battery capacity
        SoH: State of Health
        Temp: Battery temperature
        i: Flag of cycle
        Log_Charging: Save charging process data
    """

    while SoC <= SoCRange:

        if I == 0:
            break

        [_, SoC] = ECM(I, Temp, SoC)
        Temp = ThM(I, Temp, SoC)
        [Qloss, SoH] = AM(I, Temp, Qloss)

        i = i + 1

    return SoC,Qloss,SoH,Temp,i

if __name__ == '__main__':
    BatChrg([1, 1, 0, 1, 2])
