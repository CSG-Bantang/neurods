#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:00:06 2024

@author: reinierramos
"""

import numpy as np

wee = 10
wei = 12
wie = 8
wii = 3
ze = 0.2
zi = 4
tau = 1.

def odes(vars_, t, kwargs):
    """
    Set of coupled ODEs modeling Wilson-Cowan system.

    Parameters
    ----------
    vars_ : (2,) or (P, 2) array
        Values of (E, I) at time t-1.
    t : float
        Time for which vars_ is calculated.
    kwargs : dict
        Parameters defining the WC system and input current such as,
        system : str
            Type of WC system ('single', 'noisy').
        E0 : float
            Initial density of excitatory neurons.
        I0 : float
            Initial density of inhibitory neurons.
        I0_e : float
            Amplitude in uA/cm^2, of the constant current for excitatory neurons.
        I0_i : float
            Amplitude in uA/cm^2, of the constant current for inhibitory neurons.
        Is_e : float
            Amplitude in uA/cm^2, of the sinusoidal current for excitatory neurons.
        Is_i : float
            Amplitude in uA/cm^2, of the sinusoidal current for inhibitory neurons.
        fs : float
            Frequency in Hz, of the sinusoidal input for both excitatory 
            and inhibitory neurons.
        In : float
            Amplitude in uA/cm^2, of noisy input.
            Must be passed if system is 'noisy' or 'noisy coupled'.
        noise : (d,) array
            Generated random numbers from a uniform distribution [-0.5, 0.5].
            Must be passed if system is 'noisy' or 'noisy coupled'.
        noise_t : float
            Value of the noise at time t, extracted from noise.
            Must be passed if system is 'noisy' or 'noisy coupled'.

    Returns
    -------
    (2, d) or (2, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (E, I) at time t.
        
    See Also
    --------
    Iext_e : Total input stimulus current to the excitatory neurons.
    Iext_i : Total input stimulus current to the inhibitory neurons.
        
    """
    uu, vv = vars_.T
    
    I_e = Iext_e(kwargs, t)
    I_i = Iext_i(kwargs, t)
    
    dEdt = (-uu + sigmoid_function((wee * uu) - (wie * vv) - ze + I_e))/tau
    dIdt = (-vv + sigmoid_function((wei * uu) - (wii * vv) - zi + I_i))/tau
    return np.array([dEdt, dIdt])

def sigmoid_function(x:float) -> float:
    """
    Generates the sigmoid function that behaves similar to a gated channel.

    """
    return 1 / (1 + np.exp(-x))

def Iext_e(kwargs, t):
    """
    Input stimulus current to the excitatory neurons.

    Parameters
    ----------
    kwargs : dict
        Parameters defining the WC system and input current.
    t : float
        Time for which input I_e is calculated.

    Returns
    -------
    I_e : float or (P,) array
        where P = L*L
        Total input stimulus current to the WC system.

    Valid keywords in kwargs
    ------------------------
    system : str
        Type of HH system ('single', 'noisy', 'coupled', 'noisy coupled').
    I0_e : float
        Amplitude in uA/cm^2, of the constant current for excitatory neurons.
    I0_i : float
        Amplitude in uA/cm^2, of the constant current for inhibitory neurons.
    Is_e : float
        Amplitude in uA/cm^2, of the sinusoidal current for excitatory neurons.
    Is_i : float
        Amplitude in uA/cm^2, of the sinusoidal current for inhibitory neurons.
    fs : float
        Frequency in Hz, of the sinusoidal input for both excitatory 
        and inhibitory neurons.
    In : float
        Amplitude in uA/cm^2, of noisy input.
        Must be passed if system is 'noisy' or 'noisy coupled'.
    noise : (d,) array
        Generated random numbers from a uniform distribution [-0.5, 0.5].
        Must be passed if system is 'noisy' or 'noisy coupled'.
    noise_t : float
        Value of the noise at time t, extracted from noise.
        Must be passed if system is 'noisy' or 'noisy coupled'.

    """
    I0_e     = kwargs.get('I0_e')
    Is_e, fs = kwargs.get('Is_e'), kwargs.get('fs')/1000
    Isine  = Is_e * np.sin(2*np.pi*fs*t)
    if 'noisy' in kwargs.get('system'):
        sigma, eta_t = kwargs.get('In'), kwargs.get('noise_t')
        Inoise = sigma*(eta_t)
        I0_e += Inoise
    return I0_e + Isine

def Iext_i(kwargs, t):
    """
    Input stimulus current to the inhibitory neurons.

    Parameters
    ----------
    kwargs : dict
        Parameters defining the WC system and input current.
    t : float
        Time for which input I_i is calculated.

    Returns
    -------
    I_i : float or (P,) array
        where P = L*L
        Total input stimulus current to the WC system.

    Valid keywords in kwargs
    ------------------------
    system : str
        Type of HH system ('single', 'noisy', 'coupled', 'noisy coupled').
    I0_e : float
        Amplitude in uA/cm^2, of the constant current for excitatory neurons.
    I0_i : float
        Amplitude in uA/cm^2, of the constant current for inhibitory neurons.
    Is_e : float
        Amplitude in uA/cm^2, of the sinusoidal current for excitatory neurons.
    Is_i : float
        Amplitude in uA/cm^2, of the sinusoidal current for inhibitory neurons.
    fs : float
        Frequency in Hz, of the sinusoidal input for both excitatory 
        and inhibitory neurons.
    In : float
        Amplitude in uA/cm^2, of noisy input.
        Must be passed if system is 'noisy' or 'noisy coupled'.
    noise : (d,) array
        Generated random numbers from a uniform distribution [-0.5, 0.5].
        Must be passed if system is 'noisy' or 'noisy coupled'.
    noise_t : float
        Value of the noise at time t, extracted from noise.
        Must be passed if system is 'noisy' or 'noisy coupled'.

    """
    I0_i     = kwargs.get('I0_i')
    Is_i, fs = kwargs.get('Is_i'), kwargs.get('fs')/1000
    Isine  = Is_i * np.sin(2*np.pi*fs*t)
    if 'noisy' in kwargs.get('system'):
        sigma, eta_t = kwargs.get('In'), kwargs.get('noise_t')
        Inoise = sigma*(eta_t)
        I0_i += Inoise
    return I0_i + Isine