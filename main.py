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

w = 0.729 # Inertial weight
c1 = 1.49115 # Self learning factor
c2 = 1.49115 # Swarm learning factor 
beta = 1 # Weight coefficient 1: fastest; 0: healthiest
ger = 50 # The maximum number of iterations 

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
    # print(SoC, Temp)
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
    R0 = R0[0]
    R1 = griddata(Grid, R1_Tab, (SoC, temp), method='linear')
    R1 = R1[0]
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

        if I <= 0.01:
            break

        [_, SoC] = ECM(I, Temp, SoC)
        Temp = ThM(I, Temp, SoC)
        [Qloss, SoH] = AM(I, Temp, Qloss)
        i = i + 1

    return [SoC,Qloss,SoH,Temp,i]

def BatChrg(CC, iter, swarm): 
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
    SoCRange = [0.2,0.4,0.6,0.8,1.0]

    [SoC,Qloss,SoH,Temp,i] = nCC(CC[1],SoC,SoCRange[1],Qloss,SoH,Temp,i)
    [SoC,Qloss,SoH,Temp,i] = nCC(CC[2],SoC,SoCRange[2],Qloss,SoH,Temp,i)
    [SoC,Qloss,SoH,Temp,i] = nCC(CC[3],SoC,SoCRange[3],Qloss,SoH,Temp,i)
    [SoC,Qloss,SoH,Temp,i] = nCC(CC[4],SoC,SoCRange[4],Qloss,SoH,Temp,i)
    [SoC,Qloss,SoH,Temp,i] = nCC(CC[5],SoC,SoCRange[5],Qloss,SoH,Temp,i)
    t = (i - 1) * 0.05
    print("Battery Charge Process", iter, swarm)
    return [SoH,t]

def PSO(N, d, ger):
    
    iter = 1  # Initial iteration

    N = 20
    d = 5

    x = np.zeros((N,d)) # Pariticle Position (N-d)
    v = np.zeros((N,d)) # Pariticle Velcocity (N-d)

    Ilimit = np.matlib.repmat(np.array([[0],[3.3]]),1,d) # Charging current limits (2-d)
    vlimit = np.matlib.repmat(np.array([[-0.33],[0.33]]),1,d) # Velocity limits (2-d)

    # lsat = np.zeros((N,d))
    # hsat = np.zeros((N,d))

    f1 = lambda t: beta * t / t_m # Object function

    for i in range(d):
        if i == 0:
            x[:,i] = np.matlib.repmat(Ilimit[0,i],1,N) + (Ilimit[1,i] - Ilimit[0,i]) * np.random.rand(1,N)
        else:
            x[:,i] = np.matlib.repmat(Ilimit[0,i],1,N) + (x[:,i-1] - Ilimit[0,i]) * np.random.rand(1,N)
        v[:,i] = np.matlib.repmat(vlimit[0,i],1,N) + (vlimit[1,i] - vlimit[0,i]) * np.random.rand(1,N)

    xm = x # The best known position of particle (N-d)
    ym = np.zeros((1,d)) # The best known position of entire swarm (1-d)
    SoH = np.zeros((ger,N)) # SoH of each iteration of each particle (ger-N)
    t = np.zeros((ger,N)) # Charging time of each iteration of each particle (ger-N)
    fx = np.zeros((ger,N)) # Objective function value of each iteration of each particle (ger-N)
    fxm = np.zeros((N,1)) # Optimal objective function value of particle (N-1)
    fym = float("-inf") # Global optimal objective function value

    while iter <= ger:

        for j in range(N):

            [SoH[iter,j],t[iter,j]]= BatChrg(x[j],iter,j) # Battery simulation
            fx[iter,j] = f1(t[iter,j]) # Optimal function value

            if fxm[j] > fx[iter,j]:
                fxm[j] = fx[iter,j]
                xm[j] = x[j]

        if fym > np.amin(fxm):
            fym = np.amin(fxm)
            nmin = np.argmin(fxm)
            ym = xm(nmin)

        v = v * w + c1 * np.random.rand() * (xm - x) + c2 * np.random.rand() * (np.matlib.repmat(ym,N,1) - x)

        # Velcocity Saturation
        for k in range(d):
            hsat = np.where(v[:,k] < vlimit[0,k], 1, 0)
            lsat = np.where(v[:,k] > vlimit[1,k], 2, 0)

            for j in range(N):
                if hsat[j] == 1:
                    v[j,k] = vlimit[0,k]
                elif lsat[j] == 2:
                    v[j,k] = vlimit[1,k]

        x = x + v # Updating position

        # Velcocity Saturation
        for k in range(d): 
            hsat = np.where(x[:,k] < Ilimit[0,k], 1, 0)
            lsat = np.where(x[:,k] > Ilimit[1,k], 2, 0)

            for j in range(N):
                if hsat[j] == 1:
                    x[j,k] = Ilimit[0,k]
                elif lsat[j] == 2:
                    x[j,k] = Ilimit[1,k]
    
        iter = iter + 1
        print(iter)

    return [SoH,t,ym,fym,fx,fxm]

if __name__ == '__main__':
    PSO(20, 5, 50)
 