#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 22:05:20 2024

@author: reinierramos
"""

import numpy as np

"""
Bolztmann transport equations via voltage clamp model for the gated channels.
"""

def alpha_m(Phi:float) -> float: 
    """
    Activation for gating variable m of sodium channel.
    """
    return 0.1*(25-Phi) / (np.exp((25-Phi)/10)-1) 

def alpha_h(Phi:float) -> float:
    """
    Activation for gating variable h of sodium channel.
    """
    return 0.07*np.exp(-Phi/20)

def alpha_n(Phi:float) -> float:
    """
    Activation for gating variable n of potassium channel.
    """
    return 0.01*(10-Phi) / (np.exp((10-Phi)/10)-1)

def beta_m(Phi:float) -> float:
    """
    Inactivation for gating variable m of sodium channel.
    """
    return 4*np.exp(-Phi/18)

def beta_h(Phi:float) -> float:
    """
    Inactivation for gating variable h of sodium channel.
    """
    return 1 / (np.exp((30-Phi)/10)+1)

def beta_n(Phi:float) -> float:
    """
    Inactivation for gating variable n of potassium channel.
    """
    return 0.125*np.exp(-Phi/80)