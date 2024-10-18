#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 22:11:48 2024

@author: reinierramos
"""

from .bte import (alpha_m, alpha_h, alpha_n, beta_m, beta_h, beta_n)

"""
Steady-state values of gated channels m,h,n
"""

def inf_m(Phi:float=0.0) -> float:
    return alpha_m(Phi) / (alpha_m(Phi) + beta_m(Phi))
def inf_h(Phi:float=0.0) -> float:
    return alpha_h(Phi) / (alpha_h(Phi) + beta_h(Phi))
def inf_n(Phi:float=0.0) -> float:
    return alpha_n(Phi) / (alpha_n(Phi) + beta_n(Phi))

"""
Corresponding steady-state time constants of gated channels m,h,n
"""
def tau_m(Phi:float=0.0) -> float: 
    return inf_m(Phi)/alpha_m(Phi)
def tau_h(Phi:float=0.0) -> float: 
    return inf_h(Phi)/alpha_h(Phi)
def tau_n(Phi:float=0.0) -> float: 
    return inf_n(Phi)/alpha_n(Phi)