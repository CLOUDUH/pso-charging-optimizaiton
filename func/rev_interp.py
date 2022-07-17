'''
Author: CLOUDUH
Date: 2022-07-16 19:16:49
LastEditors: CLOUDUH
LastEditTime: 2022-07-16 19:39:00
Description: 
'''

import numpy as np

def rev_interp(xlist:list, ylist:list, yvalue:float):
    '''
    Args: 
        xarray: x array
        yarray: y array
        yvalue: value to be interpolated
    Returns: 
        xvalue: x value
    Detail: 
        Find the value of x corresponding to y
    '''    
    xarray = np.array(xlist)
    yarray = np.array(ylist)
    
    pos = min(np.where(yarray >= yvalue))
    xvalue = xarray[pos][0]
    
    return xvalue

if __name__ == '__main__':
    a = [1,2,3,4,5,6,7,8,9,10]
    b = [100,200,300,400,500,600,700,800,900,1000]

    print(rev_interp(a, b, 2000))