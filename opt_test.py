"""
Optimization function test 
@Author: CLOUDUH
@Data: 2022/05/28

"""

from battery_model import battery_charging

import sys
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import numpy.matlib

t_p = 1 # Step
t_m = 2*3600 # Empirical charging time
E_m = 

def optimization_test(t_p:float):
    """optimization_test

    Args:
        t_p: Step
    Returns:
        normal: CC list in paper
    """

    f = lambda SoH, E_ch: (1 - SoH) + ((E_m - E_ch) / E_m) +  # Object function