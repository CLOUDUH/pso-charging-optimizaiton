function [sys,x0,str,ts]=irridiance(t,x,u,flag)
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

t_d = u(1);
d_n = u(2);
phi = u(3);

I_SC = 1367; % 地外太阳辐射强度常数
r_0 = 149597890; % 平均日地距离常数
epislon = 1/298.256; % 地球扁率
tau = 1; % 投射系数 无量纲

% 计算单位太阳辐射强度
alpha_day = 2 * pi * (d_n - 4)/365; % 天角数
r = r_0 * (1 - epislon^2) / (1 + epislon * cos(alpha_day)); % 日地距离
I_0n = I_SC * (r_0 / r)^2; % 地外太阳辐射强度
omega = pi - (pi * t_d / 12); % 时角
delta = (23.45 * pi / 180) * sind(360 * (284 + d_n) /365); % 太阳倾斜角
theta = pi / 2 - acos(sind(phi) * sin(delta) + cosd(phi) * cos(delta) * cos(omega)); % 天顶角

P_S = max(I_0n * tau * sin(theta), 0); % 单位面积太阳辐射强度

sys=P_S;