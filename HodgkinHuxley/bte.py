#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:29:36 2024

@author: reinierramos

"""

import numpy as np

"""
Bolztmann Transport Equations
"""

def alphan(V_): return 0.01*(10-V_) / (np.exp((10-V_)/10)-1)
def alpham(V_): return 0.1*(25-V_) / (np.exp((25-V_)/10)-1) 
def alphah(V_): return 0.07*np.exp(-V_/20)
def betah(V_):  return 1 / (np.exp((30-V_)/10)+1)
def betam(V_):  return 4*np.exp(-V_/18)
def betan(V_):  return 0.125*np.exp(-V_/80)