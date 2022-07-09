'''
Author: CLOUDUH
Date: 2022-07-07 20:09:20
LastEditors: CLOUDUH
LastEditTime: 2022-07-09 23:36:43
Description: 
'''
from multiprocessing.connection import wait
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from pylab import mpl
import numpy as np

def sciplt(data:list, xlabel:str, ylabel:str, title:str, legend:str, xlim:list, ylim:list):
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

    for i in range(len(data)):
        plt.plot(data[i][0], data[i][1], label=data[i][2], marker=data[i][3], 
            color=data[i][4], markevery=data[i][5], mew=0.01)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=legend) # lower right
    plt.xlim(xlim)
    plt.ylim(ylim)

    return None



# def sciplt_display(fig, line_args:list, axes_args:list, title:str):
#     '''display set
#     Args:
#         fig: figure
#         line_args: args of display [[221],[x1,x2,x3], [y1,y2,y3], [label,color,marker,linewidth,markevery]]
#         axes_args: args of axes [[]]

        
#     Returns:
#         None
#     '''

#     mpl.rcParams['font.sans-serif'] = ['Times New Roman'] # 指定默认字体
#     mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

#     plt.rcParams['figure.figsize'] = (6.0, 4.0) # 设置 figure_size
#     plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
#     plt.rcParams['image.cmap'] = 'gray' # 设置 颜色 style
#     plt.rcParams['savefig.dpi'] = 300 #图片像素
#     plt.rcParams['figure.dpi'] = 300 #分辨率

#     plt.grid() 

#     ax_list = {}
#     pst_list = []

#     for i in range(len(line_args)):
#         pst_list.append(line_args[i][0][0])

#     pst_list = list(set(pst_list))   

#     for pst in pst_list:
#         ax_list[pst] = fig.add_subplot(pst)

#     for i in range(len(line_args)):
#         line = Line2D(line_args[i][1], line_args[i][2])
#         line.set(label=line_args[i][3][0], color=line_args[i][3][1], marker=line_args[i][3][2], 
#             linewidth=line_args[i][3][3], markevery=line_args[i][3][4], mew=0.01)
#         ax_list[line_args[i][0][0]].add_line(line)

#     i = 0
#     for ax in ax_list.values():
#         ax.set(xlabel=axes_args[i][0], ylabel=axes_args[i][1], title=axes_args[i][2])
#         ax.legend(loc=axes_args[i][3])
#         i += 1

#     plt.show()

if __name__ == '__main__':
    a = np.linspace(0,10,100)
    b = 0.8*a
    c = 1.2*a
    u = np.sin(a)
    v = np.cos(b)
    w = u+v
    
    plt.show()