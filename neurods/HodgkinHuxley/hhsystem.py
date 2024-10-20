#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 22:14:42 2024

@author: reinierramos
"""

import numpy as np
from typing import Final
from .bte import (alpha_m, alpha_h, alpha_n, beta_m, beta_h, beta_n)

C_mem  : Final =    1     ## Membrane capacitance,         in $ \mu F / cm^2 $
G_Na   : Final =  120     ## Sodium conductance,           in $ mS / cm^2 $
G_K    : Final =   36     ## Potassium conductance,        in $ mS / cm^2 $
G_leak : Final =    0.3   ## Leakage conductance,          in $ mS / cm^2 $
E_Na   : Final =  115     ## Sodium reversal potential,    in $ mV $
E_K    : Final = - 12     ## Potassium reversal potential, in $ mV $
E_leak : Final =   10.6   ## Leakage reversal potential,   in $ mV $

def odes(vars_, t, kwargs):
    """
    Set of coupled ODEs modeling Hodgkin-Huxley system.

    Parameters
    ----------
    vars_ : (4,) or (P, 4) array
        Values of (Phi, m, h, n) at time t-1.
    t : float
        Time for which vars_ is calculated.
    kwargs : dict
        Parameters defining the HH system and input current such as,
        system : str
            Type of HH system ('single', 'noisy', 'coupled', 'noisy coupled').
        Vrest : float
            Resting voltage.
        I0 : float
            Amplitude in uA/cm^2, of the constant or bias current.
        Is : float
            Amplitude in uA/cm^2, of the sine input.
        fs : float
            Frequency in Hz, of the sine input.
        In : float
            Amplitude in uA/cm^2, of noisy input.
            Must be passed if system is 'noisy' or 'noisy coupled'.
        L : int
            Lattice size of the HH network.
            Must be passed if system is 'coupled' or 'noisy coupled'.
        g : float
            Uniform coupling strength between neighboring neurons in 
            the lattice. Must be passed if system is 'coupled' or 
            'noisy coupled'.
        noise : (d,) array
            Generated random numbers from a uniform distribution [-0.5, 0.5].
            Must be passed if system is 'noisy' or 'noisy coupled'.
        noise_t : float
            Value of the noise at time t, extracted from noise.
            Must be passed if system is 'noisy' or 'noisy coupled'.
        aij : (P,P) array
            Adjacency matrix of the square lattice.
            Must be passed if system is 'coupled' or 'noisy coupled'.
        Vij : (P,P) array
            Voltage difference Vi - Vj from neighboring neurons.
            Must be passed if system is 'coupled' or 'noisy coupled'.

    Returns
    -------
    (4, d) or (4, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (Phi, m, h, n) at time t.
        
    See Also
    --------
    Iext : Total input stimulus current to the HH system.
        
    """
    Phi, m, h, n   = vars_.T
    
    current_Na   = G_Na   * (E_Na  -Phi) * np.power(m,3) * h 
    current_K    = G_K    * (E_K   -Phi) * np.power(n,4)
    current_leak = G_leak * (E_leak-Phi)
    
    I = Iext(kwargs, t)  ## total input current

    dPhidt = (current_Na + current_K + current_leak + I) / C_mem
    dmdt   = alpha_m(Phi)*(1-m) - beta_m(Phi)*m
    dhdt   = alpha_h(Phi)*(1-h) - beta_h(Phi)*h
    dndt   = alpha_n(Phi)*(1-n) - beta_n(Phi)*n
    return np.array([dPhidt, dmdt, dhdt, dndt])

def Iext(kwargs, t):
    """
    Input stimulus current to a single HH neuron.

    Parameters
    ----------
    kwargs : dict
        Parameters defining the HH system and input current.
    t : float
        Time for which input I is calculated.

    Returns
    -------
    I : float or (P,) array
        where P = L*L
        Total input stimulus current to the HH system.

    Valid keywords in kwargs
    ------------------------
    system : str
        Type of HH system ('single', 'noisy', 'coupled', 'noisy coupled').
    Vrest : float
        Resting voltage.
    I0 : float
        Amplitude in uA/cm^2, of the constant or bias current.
    Is : float
        Amplitude in uA/cm^2, of the sine input.
    fs : float
        Frequency in Hz, of the sine input.
    In : float
        Amplitude in uA/cm^2, of noisy input.
        Must be passed if system is 'noisy' or 'noisy coupled'.
    L : int
        Lattice size of the HH network.
        Must be passed if system is 'coupled' or 'noisy coupled'.
    g : float
        Uniform coupling strength between neighboring neurons in the lattice.
        Must be passed if system is 'coupled' or 'noisy coupled'.
    noise : (d,) array
        Generated random numbers from a uniform distribution [-0.5, 0.5].
        Must be passed if system is 'noisy' or 'noisy coupled'.
    noise_t : float
        Value of the noise at time t, extracted from noise.
        Must be passed if system is 'noisy' or 'noisy coupled'.
    aij : (P,P) array
        Adjacency matrix of the square lattice.
        Must be passed if system is 'coupled' or 'noisy coupled'.
    Vij : (P,P) array
        Voltage difference Vi - Vj from neighboring neurons.
        Must be passed if system is 'coupled' or 'noisy coupled'.

    """
    I0     = kwargs.get('I0')
    Is, fs = kwargs.get('Is'), kwargs.get('fs')/1000
    Isine  = Is * np.sin(2*np.pi*fs*t)
    if 'noisy' in kwargs.get('system'):
        sigma, eta_t = kwargs.get('In'), kwargs.get('noise_t')
        Inoise = sigma*(eta_t)
        I0 += Inoise
    if 'coupled' in kwargs.get('system'):
        g, aij, Vij = kwargs.get('g'), kwargs.get('aij'), kwargs.get('Vij')
        Iij = np.sum(-g*aij*Vij, axis=0)
        Iij[0] += I0 + Isine
        return Iij
    return I0 + Isine
