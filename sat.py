'''
Author: CLOUDUH
Date: 2022-06-03 11:38:58
LastEditors: CLOUDUH
LastEditTime: 2022-06-03 11:43:23
Description: Saturation Funtion
'''

from math import inf

def sat(value:float, llim:float, hlim:float):
    '''Saturation Function
    Args: 
        value: input value
        llim: low limit
        hlim: high limit
    Returns: 
        value: after procese
    Detail: 
        you can input "inf" or "-inf" to make this function
        to be hsat or lsat
    '''    

    if value < llim:
        value = llim
    
    if value > hlim:
        value = hlim

    return value

if __name__ == '__main__':
    print(sat(10, -1, 2))
    print(sat(10, -1, inf))