#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:34:54 2024

@author: reinierramos
"""

from .bte import (alphah, alpham, alphan, betah, betam, betan)

### Steady-state values of gated channels
def n_inf(V_=0.0):  return alphan(V_) / (alphan(V_) + betan(V_))
def m_inf(V_=0.0):  return alpham(V_) / (alpham(V_) + betam(V_))
def h_inf(V_=0.0):  return alphah(V_) / (alphah(V_) + betah(V_))

### Corresponding Steady-state time constants
def tau_m(V_=0.0):      return m_inf(V_)/alpham(V_)
def tau_h(V_=0.0):      return h_inf(V_)/alphah(V_)
def tau_n(V_=0.0):      return n_inf(V_)/alphan(V_)