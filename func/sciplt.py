'''
Author: CLOUDUH
Date: 2022-07-07 20:09:20
LastEditors: CLOUDUH
LastEditTime: 2022-07-07 22:22:58
Description: 
'''
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np

def sciplt_pso(x:list, y:dict, xlabel:str, ylabel:str, title:str):
    '''plot
    Args:
        x: x-axis
        y: y-axis
        xlabel: x-axis labe
        ylabel: y-axis label
        title: title of plot
        style: style of plot (0-15)
    Returns:
        None
    '''

    data = []
    name = []
    N = 0

    for key, value in y.items():
        data.append(value)
        name.append(key)
        N += 1

    marker_preset = ["o","v","^","<",">","s","p","*"]
    color_preset = ["r","g","b","c","m","y","k","w"]

    mpl.rcParams['font.sans-serif'] = ['Times New Roman'] # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

    plt.rcParams['figure.figsize'] = (6.0, 4.0) # 设置 figure_size
    plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
    plt.rcParams['image.cmap'] = 'gray' # 设置 颜色 style
    plt.rcParams['savefig.dpi'] = 300 #图片像素
    plt.rcParams['figure.dpi'] = 300 #分辨率

    plt.grid()
    for i in range(N):
        plt.plot(x, data[i], marker=marker_preset[i], color=color_preset[i], mew=0.1, label=name[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower left")

    return None

def sciplt_battery(x:list, y:dict, xlabel:str, ylabel:str, title:str, name:list):
    '''plot
    Args:
        x: x-axis
        y: y-axis
        xlabel: x-axis labe
        ylabel: y-axis label
        title: title of plot
        style: style of plot (0-15)
    Returns:
        None
    '''

    marker_preset = ["o","v","^","<",">","s","p","*"]
    color_preset = ["r","g","b","c","m","y","k","w"]

    mpl.rcParams['font.sans-serif'] = ['Times New Roman'] # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

    plt.rcParams['figure.figsize'] = (6.0, 4.0) # 设置 figure_size
    plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
    plt.rcParams['image.cmap'] = 'gray' # 设置 颜色 style
    plt.rcParams['savefig.dpi'] = 300 #图片像素
    plt.rcParams['figure.dpi'] = 300 #分辨率

    plt.grid()
    for i in range(len(x)):
        plt.plot(x[i], y[i], marker=marker_preset[i], color=color_preset[i], markevery=1000, mew=0.01, label=name[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower left")

    return None

if __name__ == '__main__':
    a = np.linspace(0,10,100)
    b = 0.8*a
    c = 1.2*a
    u = np.sin(a)
    v = np.cos(b)
    w = u+v

    x = a
    y = {"u": u, "v": v, "w": w}

    # sciplt(a,u,"position","value","sin(x)",6)
    sciplt_(x,y,"position","value","cos(x)")

    plt.show()