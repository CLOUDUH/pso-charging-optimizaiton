function [sys,x0,str,ts]=mppt(t,x,u,flag)
switch flag
    case 0,
        [sys,x0,str,ts]=mdlInitializeSizes;
    case 2,
        sys=mdlUpdate(t,x,u);
    case 3,
        sys=mdlOutputs(t,x,u);
    case 9,
        sys=[];
    otherwise 
        error(['Unhandled flag=',num2str(flag)]);    
end

function [sys,x0,str,ts]=mdlInitializeSizes(len)
sizes=simsizes;%返回一个变量，
sizes.NumContStates=0;%连续变量个数
sizes.NumDiscStates=1;%离散变量个数
sizes.NumOutputs=1;%输出个数
sizes.NumInputs=3;%输入个数
sizes.DirFeedthrough=1;%直接贯通，0或1，当时输出值直接依赖于同一时刻的输入时为1
sizes.NumSampleTimes=1;%采样时间
sys=simsizes(sizes);%返回值
x0=0;
str=[];
ts=[1 0];

function sys=mdlUpdate(t,x,u)
sys=[];

function sys=mdlOutputs(t,x,u)

% 输入量
irradiation = u(1);
temp = u(2);
vol_out = u(3);

% 通过四个参数设定光伏曲线
vol_oc_ref = 60;
vol_mp_ref = 50;
cur_sc_ref = 3.6;
cur_mp_ref = 3;

% 通过温度对三个参数进行修正
vol_oc = vol_oc_ref * log(exp(1) + 0.0005 * (abs(irradiation - 1000))) * (1 - 0.00288 * (abs(temp - 25)))
vol_mp = vol_mp_ref * log(exp(1)+0.0005 * (abs(irradiation - 1000))) * (1 - 0.00288 * (abs(temp - 25)))
cur_sc = cur_sc_ref * (irradiation / 1000) * (1 + 0.0025 * (abs(temp - 25)))
cur_mp = cur_mp_ref * (irradiation / 1000)*(1 + 0.0025 * (abs(temp - 25)))

% 计算在此电压下的光伏电流
% c2 = ((vol_mp / vol_oc) - 1) * (log(1 - (cur_mp / cur_sc)))^(-1)
% c1 = (1 - (cur_mp / cur_sc)) * exp(-vol_mp / (c2 * vol_oc))
% cur = cur_sc *( 1-c1 *( exp( vol / (c2 * vol_oc))-1))

% 计算最大功率点下的电流（一般是电池）
cur_out = vol_mp * cur_mp / vol_out

sys=cur_out;