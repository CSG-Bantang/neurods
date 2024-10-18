#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 21:13:49 2024

@author: reinierramos
"""

import numpy as np
from typing import Final

V1     : Final = -  1.2
V2     : Final =   18
V3     : Final =   12
V4     : Final =   17.4
phi    : Final =    0.067

"""
Steady-state values of gated channels m, n
"""

def inf_m(Phi):
    return 0.5*(1+np.tanh((Phi-V1)/V2))

def inf_n(Phi):
    return 0.5*(1+np.tanh((Phi-V3)/V4))

"""
Corresponding steady-state time constants of gated channel n
"""
def tau_n(Phi):
    return 1 / (phi * np.cosh((Phi-V3)/(2*V4)))