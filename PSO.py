import numpy as np
import pandas as pd
import numpy.matlib
Battery('PSO','Charging')
Initialization
clear
global t_p
global SoC_Grid
global Temp_Grid
global VO_Tab
global R0_Tab
global tau1_Tab
global R1_Tab
global Tf
global m
global A
global c
global h
global z
global B
global E_a
global R
global alpha
global Qe

# 
VO_Tab = pd.read_csv("utils/VO.csv",header=None)
R0_Tab = pd.read_csv("utils/R0.csv",header=None)
R1_Tab = pd.read_csv("utils/R1.csv",header=None)
tau1_Tab = pd.read_csv("utils/tau1.csv",header=None)
Temp_Grid = pd.read_csv("utils/Temp_Grid.csv",header=None)
SoC_Grid = pd.read_csv("utils/SoC_Grid.csv",header=None)

t_p = 1
Tf = 298.15
Qe = 3.3
t_m = 2 * 3600
h = 40.106
c = 800
A = 0.004317
m = 0.0475
z = 0.4
B = 130
E_a = 18461
R = 8.314

alpha = 32

PSO('Method')
N = 5

d = 5

x = np.zeros((N,d))

v = np.zeros((N,d))

iter = 1

ger = 50

Ilimit = np.matlib.repmat(np.array([[0],[3.3]]),1,d)

vlimit = np.matlib.repmat(np.array([[- 0.33],[0.33]]),1,d)

lsat = np.zeros((N,d))
hsat = np.zeros((N,d))
w = 0.729

c1 = 1.49115

c2 = 1.49115

beta = 1

# f= @(SoH, t)(1 - beta) * (1 - SoH) + beta * t / t_m; # Objective function
f1 = lambda t = None: beta * t / t_m

for i in np.arange(1,d+1).reshape(-1):
    if i == 1:
        x[:,i] = np.matlib.repmat(Ilimit(1,i),N,1) + (Ilimit(2,i) - Ilimit(1,i)) * np.random.rand(N,1)
    else:
        x[:,i] = np.matlib.repmat(Ilimit(1,i),N,1) + (x(i - 1) - Ilimit(1,i)) * np.random.rand(N,1)
    v[:,i] = np.matlib.repmat(vlimit(1,i),N,1) + (vlimit(2,i) - vlimit(1,i)) * np.random.rand(N,1)

x = gpuArray(x)
v = gpuArray(v)
xm = x

ym = np.zeros((1,d))

SoH = np.zeros((ger,N))

t = np.zeros((ger,N))

fx = np.zeros((ger,N))

fxm = np.zeros((N,1))

fym = - inf

while iter <= ger:

    for j in np.arange(1,N+1).reshape(-1):
        SoH[iter,j],t[iter,j] = BatChrg(x(j,:))
        # fx(iter, j) = f(SoH(iter, j), t(iter,j)) # Optimal function value
        fx[iter,j] = f1(t(iter,j))
        if fxm(j) > fx(iter,j):
            fxm[j] = fx(iter,j)
            xm[j,:] = x(j,:)
    if fym > np.amin(fxm):
        fym,nmin = np.amin(fxm)
        ym = xm(nmin,:)
    v = v * w + c1 * rand * (xm - x) + c2 * rand * (np.matlib.repmat(ym,N,1) - x)
    for k in np.arange(1,d+1).reshape(-1):
        v[v[:,k] < vlimit[1,k]] = vlimit(1,k)
        v[v[:,k] > vlimit[2,k]] = vlimit(2,k)
    x = x + v
    for k in np.arange(1,d+1).reshape(-1):
        x[x[:,k] < Ilimit[1,k]] = Ilimit(1,k)
        x[x[:,k] > Ilimit[2,k]] = Ilimit(2,k)
    x
    iter = iter + 1


Battery('Model','Function')

def ECM(I = None,Temp = None,SoC = None): 
# Battery 1-RC Equivalent Circuit Model
# Input:
#   I: Battery current(Charging positive)
#   Temp: Battery temperature
#   SoC: State of Charge
# Output:
#   Vt: Battery Voltage
#   SoC: State of Charge

global t_p
global SoC_Grid
global Temp_Grid
global VO_Tab
global R0_Tab
global tau1_Tab
global Qe
temp = Temp - 273.15
VO = interp2(Temp_Grid,SoC_Grid,VO_Tab,temp,SoC)
R0 = interp2(Temp_Grid,SoC_Grid,R0_Tab,temp,SoC)
tau1 = interp2(Temp_Grid,SoC_Grid,tau1_Tab,temp,SoC)
Vt = VO + I * (1 - np.exp(t_p / tau1)) + I * R0
SoC = SoC + t_p * I / (3600 * Qe)
return Vt,SoC


def ThM(I = None,Temp = None,SoC = None): 
# Battery Thermal Model
# Input:
#   I: Battery current(Charging positive)
#   Temp: Battery temperature
#   SoC: State of Charge
# Output:
#   Temp: Battery temperature

global t_p
global SoC_Grid
global Temp_Grid
global R0_Tab
global R1_Tab
global Tf
global m
global A
global c
global h
temp = Temp - 273.15
R0 = interp2(Temp_Grid,SoC_Grid,R0_Tab,temp,SoC)
R1 = interp2(Temp_Grid,SoC_Grid,R1_Tab,temp,SoC)
dVO_T = (13.02 * SoC ** 5 - 38.54 * SoC ** 4 + 32.81 * SoC ** 3 + 2.042 * SoC ** 2 - 13.93 * SoC + 4.8) / 1000
dTemp = (I ** 2 * (R0 + R1) + I * Temp * dVO_T - h * A * (Temp - Tf)) / (m * c)
Temp = Temp + dTemp * t_p
return Temp


def AM(I = None,Temp = None,Qloss = None): 
# Battery Aging Model
# Input:
#   I: Battery current(Charging positive)
#   Temp: Battery temperature
#   Qloss: Loss battery capacity
# Output:
#   Qloss: Loss battery capacity
#   SoH: State of Health

global t_p
global z
global B
global E_a
global R
global alpha
global Qe
dQloss = (np.abs(I) / 3600) * z * B * np.exp((- E_a + alpha * np.abs(I)) / (R * Temp)) * (Qloss / (B * np.exp((- E_a + alpha * np.abs(I)) / (R * Temp)))) ** (1 - (1 / z))
Qloss = Qloss + dQloss * t_p
SoH = 1 - ((Qloss / Qe) / 0.2)
return Qloss,SoH

# function [SoH, t, Log] = BatChrg(CC)

def BatChrg(CC = None): 
# Battery Charging Function
# Input:
#   CC: Battery charging constant-current(5-1 matrix)
# Output:
#   SoH: Whole charging process SoH
#   t: Charging time cost

global Qe
SoC = 0.1

Qloss = 0.0001

SoH = 1 - ((Qloss / Qe) / 0.2)

Temp = 298.15

i = 1

SoCRange = np.array([0.2,0.4,0.6,0.8,1.0])
#     tic
#     [SoC, Qloss, SoH, Temp, i, Log_CC1] = nCC(CC(1), SoC, SoCRange(1), Qloss, SoH, Temp, i);
#     toc
#     tic
#     [SoC, Qloss, SoH, Temp, i, Log_CC2] = nCC(CC(2), SoC, SoCRange(2), Qloss, SoH, Temp, i);
#     toc
#     tic
#     [SoC, Qloss, SoH, Temp, i, Log_CC3] = nCC(CC(3), SoC, SoCRange(3), Qloss, SoH, Temp, i);
#     toc
#     tic
#     [SoC, Qloss, SoH, Temp, i, Log_CC4] = nCC(CC(4), SoC, SoCRange(4), Qloss, SoH, Temp, i);
#     toc
#     tic
#     [SoC, Qloss, SoH, Temp, i, Log_CC5] = nCC(CC(5), SoC, SoCRange(5), Qloss, SoH, Temp, i);
#     toc

SoC,Qloss,SoH,Temp,i = nCC(CC(1),SoC,SoCRange(1),Qloss,SoH,Temp,i)
SoC,Qloss,SoH,Temp,i = nCC(CC(2),SoC,SoCRange(2),Qloss,SoH,Temp,i)
SoC,Qloss,SoH,Temp,i = nCC(CC(3),SoC,SoCRange(3),Qloss,SoH,Temp,i)
SoC,Qloss,SoH,Temp,i = nCC(CC(4),SoC,SoCRange(4),Qloss,SoH,Temp,i)
SoC,Qloss,SoH,Temp,i = nCC(CC(5),SoC,SoCRange(5),Qloss,SoH,Temp,i)
# Log = [Log_CC1; Log_CC2; Log_CC3; Log_CC4; Log_CC5];
t = (i - 1) * 0.05
return SoH,t

# function [SoC, Qloss, SoH, Temp, i, Log_Charging] = nCC(I, SoC, SoCRange, Qloss, SoH, Temp, i)

def nCC(I = None,SoC = None,SoCRange = None,Qloss = None,SoH = None,Temp = None,i = None): 
# Battery n-Constant Current Charging Process
# Input:
#   I: Battery current(Charging positive)
#   SoC: State of Charge
#   SoCRange: n-CC Charging SoC Range
#   Qloss: Loss battery capacity
#   SoH: State of Health
#   Temp: Battery temperature
#   i: Flag of cycle
# Output:
#   SoC: State of Charge
#   Qloss: Loss battery capacity
#   SoH: State of Health
#   Temp: Battery temperature
#   i: Flag of cycle
#   Log_Charging: Save charging process data

while SoC <= SoCRange:

    if I == 0:
        break
    # [Vt, SoC] = ECM(I, Temp, SoC);
    __,SoC = ECM(I,Temp,SoC)
    Temp = ThM(I,Temp,SoC)
    Qloss,SoH = AM(I,Temp,Qloss)
    #         # record
#         Log_Charging = zeros(20000,7); # Preallocated memory
#         Log_Charging(i, 1) = (i - 1) * 0.05;
#         Log_Charging(i, 2) = Vt;
#         Log_Charging(i, 3) = I;
#         Log_Charging(i, 4) = SoC;
#         Log_Charging(i, 5) = Qloss;
#         Log_Charging(i, 6) = SoH;
#         Log_Charging(i, 7) = Temp;
    i = i + 1


return SoC,Qloss,SoH,Temp,i
