#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 21:03:10 2024

@author: reinierramos
"""

import numpy as np
from typing import Final
from .mlsteadystate import (inf_m, inf_n, tau_n)

C_mem  : Final =   20     ## Membrane capacitance,         in $ \mu F / cm^2 $
G_Ca   : Final =    4     ## Sodium conductance,           in $ mS / cm^2 $
G_K    : Final =    8     ## Potassium conductance,        in $ mS / cm^2 $
G_leak : Final =    2     ## Leakage conductance,          in $ mS / cm^2 $
E_Ca   : Final =  120     ## Sodium reversal potential,    in $ mV $
E_K    : Final = - 84     ## Potassium reversal potential, in $ mV $
E_leak : Final = - 60     ## Leakage reversal potential,   in $ mV $

def odes(vars_, t, kwargs):
    """
    Set of coupled ODEs modeling Hodgkin-Huxley system.

    Parameters
    ----------
    vars_ : (2,) or (P, 2) array
        Values of (Phi, n) at time t-1.
    t : float
        Time for which vars_ is calculated.
    kwargs : dict
        Parameters defining the ML system and input current such as,
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
    (2, d) or (2, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (Phi, n) at time t.
        
    See Also
    --------
    Iext : Total input stimulus current to the ML system.
        
    """
    Phi, n   = vars_.T
    
    current_Ca   = G_Ca   * (E_Ca  -Phi) * inf_m(Phi)
    current_K    = G_K    * (E_K   -Phi) * n
    current_leak = G_leak * (E_leak-Phi)
    
    I = Iext(kwargs, t)  ## total input current

    dPhidt = (current_Ca + current_K + current_leak + I) / C_mem
    dndt   = (inf_n(Phi) - n) / tau_n(Phi)
    return np.array([dPhidt, dndt])

def Iext(kwargs, t):
    """
    Input stimulus current to a single HH neuron.

    Parameters
    ----------
    kwargs : dict
        Parameters defining the ML system and input current such as,
    t : float
        Time for which input I is calculated.

    Returns
    -------
    I : float or (P,) array
        where P = L*L
        Total input stimulus current to the ML system.

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