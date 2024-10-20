#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:14:40 2024

@author: reinierramos
"""

import numpy as np
from scipy.integrate import odeint
from .wcsystem import (odes, Iext_e, Iext_i)

def lsoda(tList, kwargs):
    """
    Solve the ODEs using LSODA (Livermore Solver for Ordinary Differential
    Equations) via the implementation of `scipy.integrate.odeint`).

    Parameters
    ----------
    tList : (d,) array
        Time values to solve the ODE.
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

    Returns
    -------
    soln : (2, d) or (2, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (E, I) for all t in tList.
    I_elist : (d,) or (P, d) array
        Total input current for excitatory neurons.
    I_ilist : (d,) or (P, d) array
        Total input current for inhibitory neurons.
    
    """
    E0 = kwargs.get('E0')
    I0 = kwargs.get('I0')
    guess = np.array([E0, I0])
    soln  = odeint(odes, guess, tList, args=(kwargs,))
    I_elist = np.array([Iext_e(kwargs, t) for t in tList])
    I_ilist = np.array([Iext_i(kwargs, t) for t in tList])
    return soln, I_elist, I_ilist

def euler(tList, kwargs):
    """
    Solve the ODEs using the forward Euler method.

    Parameters
    ----------
    tList : (d,) array
        List of time values to solve the ODE.
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

    Returns
    -------
    soln : (2, d) or (2, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (E, I) for all t in tList.
    I_elist : (d,) or (P, d) array
        Total input current for excitatory neurons.
    I_ilist : (d,) or (P, d) array
        Total input current for inhibitory neurons.
        
    """
    dt = tList[1] - tList[0]
    E0 = kwargs.get('E0')
    I0 = kwargs.get('I0')
    guess = np.array([E0, I0])
    
    if 'noisy' in kwargs.get('system'):
        noise_ = kwargs.get('noise')[0]
        kwargs.update({'noise_t': noise_})
      
    soln    = np.zeros([len(tList), 2])
    soln[0] = guess
    I_elist = np.zeros(len(tList))
    I_elist[0] = Iext_e(kwargs, tList[0])
    I_ilist = np.zeros(len(tList))
    I_ilist[0] = Iext_i(kwargs, tList[0])
    
    for _i in range(len(tList)-1):
        if 'noisy' in kwargs.get('system'):
            noise_ = kwargs.get('noise')[_i]
            kwargs.update({'noise_t': noise_})
        
        next_ = odes(guess, tList[_i+1], kwargs).T
        guess += next_*dt
        soln[_i+1] = guess
        I_elist[_i+1] = Iext_e(kwargs, tList[_i+1])
        I_ilist[_i+1] = Iext_i(kwargs, tList[_i+1])
    return soln, I_elist, I_ilist

def rk4(tList, kwargs):
    """
    Solve the ODEs using the Runge-Kutta 4th order method.

    Parameters
    ----------
    tList : (d,) array
        List of time values to solve the ODE.
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

    Returns
    -------
    soln : (2, d) or (2, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (E, I) for all t in tList.
    I_elist : (d,) or (P, d) array
        Total input current for excitatory neurons.
    I_ilist : (d,) or (P, d) array
        Total input current for inhibitory neurons.
    
    """
    dt = tList[1] - tList[0]
    E0 = kwargs.get('E0')
    I0 = kwargs.get('I0')
    guess = np.array([E0, I0])
    
    if 'noisy' in kwargs.get('system'):
        noise_ = kwargs.get('noise')[0]
        kwargs.update({'noise_t': noise_})
      
    soln    = np.zeros([len(tList), 2])
    soln[0] = guess
    I_elist = np.zeros(len(tList))
    I_elist[0] = Iext_e(kwargs, tList[0])
    I_ilist = np.zeros(len(tList))
    I_ilist[0] = Iext_i(kwargs, tList[0])
        
    for _i in range(len(tList)-1):
        if 'noisy' in kwargs.get('system'):
            noise_ = kwargs.get('noise')[_i]
            kwargs.update({'noise_t': noise_})
        k1 = dt * odes(guess,        tList[_i+1],        kwargs).T
        k2 = dt * odes(guess+0.5*k1, tList[_i+1]+0.5*dt, kwargs).T
        k3 = dt * odes(guess+0.5*k2, tList[_i+1]+0.5*dt, kwargs).T
        k4 = dt * odes(guess+k3,     tList[_i+1]+dt,     kwargs).T
        guess += (k1 + 2*(k2+k3) + k4)/6
        soln[_i+1] = guess
        I_elist[_i+1] = Iext_e(kwargs, tList[_i+1])
        I_ilist[_i+1] = Iext_i(kwargs, tList[_i+1])
    return soln, I_elist, I_ilist