#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:28:58 2024

@author: reinierramos
"""

import numpy as np
from scipy.integrate import odeint
from .steadystate import (inf_m, inf_h, inf_n)
from .hhsystem import (odes, Iext)

def lsoda(tList, kwargs):
    """
    Solve the ODEs using LSODA (Livermore Solver for Ordinary Differential
    Equations) via the implementation of `scipy.integrate.odeint`).

    Parameters
    ----------
    tList : (d,) array
        Time values to solve the ODE.
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

    Returns
    -------
    soln : (4, d) or (4, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (Phi, m, h, n) for all t in tList.
    Ilist : (d,) or (P, d) array
        Total input current for all t in tlist.
    
    """
    Vrest = kwargs.get('Vrest')
    guess = np.array([Vrest, inf_m(Vrest), inf_h(Vrest), inf_n(Vrest)])
    soln  = odeint(odes, guess, tList, args=(kwargs,))
    Ilist = np.array([Iext(kwargs, t) for t in tList])
    return soln, Ilist

def euler(tList, kwargs):
    """
    Solve the ODEs using the forward Euler method.

    Parameters
    ----------
    tList : (d,) array
        List of time values to solve the ODE.
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
        aij : (P,P) array
            Adjacency matrix of the square lattice.
            Must be passed if system is 'coupled' or 'noisy coupled'.

    Returns
    -------
    soln : (4, d) or (4, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (Phi, m, h, n) for all t in tList.
    Ilist : (d,) or (P, d) array
        Total input current for all t in tlist.
        
    """
    dt = tList[1] - tList[0]
    Vrest = kwargs.get('Vrest')
    guess = np.array([Vrest, inf_m(Vrest), inf_h(Vrest), inf_n(Vrest)])
    
    if 'noisy' in kwargs.get('system'):
        noise_ = kwargs.get('noise')[0]
        kwargs.update({'noise_t': noise_})
    if 'coupled' in kwargs.get('system'):
        population = np.power(kwargs.get('L'), 2)
        soln = np.zeros([population, len(tList), 4])
        soln[:,0,:] = guess
        guess = soln[:,0,:]
        
        Vi = np.tile(soln[:,0,0], (population,1))
        Vj = Vi.T
        Vij = Vi-Vj
        kwargs.update({'Vij':Vij})
        Ilist = np.zeros([population, len(tList)])
        Ilist[:,0] = Iext(kwargs, 0)
    else:
        soln    = np.zeros([len(tList), 4])
        soln[0] = guess
        Ilist = np.zeros(len(tList))
        Ilist[0] = Iext(kwargs, tList[0])
    
    for _i in range(len(tList)-1):
        if 'noisy' in kwargs.get('system'):
            noise_ = kwargs.get('noise')[_i]
            kwargs.update({'noise_t': noise_})
        if 'coupled' in kwargs.get('system'):
            Vi = np.tile(soln[:,_i,0], (population,1))
            Vj = Vi.T
            Vij = Vi-Vj
            kwargs.update({'Vij':Vij})
        next_ = odes(guess, tList[_i+1], kwargs).T
        guess += next_*dt
        if 'coupled' in kwargs.get('system'): 
            soln[:,_i+1, :] = guess
            Ilist[:,_i+1] = Iext(kwargs, tList[_i+1])
        else: 
            soln[_i+1] = guess
            Ilist[_i+1] = Iext(kwargs, tList[_i+1])
    return soln, Ilist

def rk4(tList, kwargs):
    """
    Solve the ODEs using the Runge-Kutta 4th order method.

    Parameters
    ----------
    tList : (d,) array
        List of time values to solve the ODE.
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
        aij : (P,P) array
            Adjacency matrix of the square lattice.
            Must be passed if system is 'coupled' or 'noisy coupled'.

    Returns
    -------
    soln : (4, d) or (4, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (Phi, m, h, n) for all t in tList.
    Ilist : (d,) or (P, d) array
        Total input current for all t in tlist.
    
    """
    dt = tList[1] - tList[0]
    Vrest = kwargs.get('Vrest')
    guess = np.array([Vrest, inf_m(Vrest), inf_h(Vrest), inf_n(Vrest)])
    if 'noisy' in kwargs.get('system'):
        noise_ = kwargs.get('noise')[0]
        kwargs.update({'noise_t': noise_})
    if 'coupled' in kwargs.get('system'):
        population = np.power(kwargs.get('L'), 2)
        soln = np.zeros([population, len(tList), 4])
        soln[:,0,:] = guess
        guess = soln[:,0,:]
        
        Vi = np.tile(soln[:,0,0], (population,1))
        Vj = Vi.T
        Vij = Vi-Vj
        kwargs.update({'Vij':Vij})
        Ilist = np.zeros([population, len(tList)])
        Ilist[:,0] = Iext(kwargs, 0)
    else:
        soln    = np.zeros([len(tList), 4])
        soln[0] = guess
        Ilist = np.zeros(len(tList))
        Ilist[0] = Iext(kwargs, tList[0])
        
    for _i in range(len(tList)-1):
        if 'noisy' in kwargs.get('system'):
            noise_ = kwargs.get('noise')[_i]
            kwargs.update({'noise_t': noise_})
        if 'coupled' in kwargs.get('system'):
            Vi = np.tile(soln[:,_i,0], (population,1))
            Vj = Vi.T
            Vij = Vi-Vj
            kwargs.update({'Vij':Vij})
        k1 = dt * odes(guess,        tList[_i+1],        kwargs).T
        k2 = dt * odes(guess+0.5*k1, tList[_i+1]+0.5*dt, kwargs).T
        k3 = dt * odes(guess+0.5*k2, tList[_i+1]+0.5*dt, kwargs).T
        k4 = dt * odes(guess+k3,     tList[_i+1]+dt,     kwargs).T
        guess += (k1 + 2*(k2+k3) + k4)/6
        if 'coupled' in kwargs.get('system'): 
            soln[:,_i+1, :] = guess
            Ilist[:,_i+1] = Iext(kwargs, tList[_i+1])
        else: 
            soln[_i+1] = guess
            Ilist[_i+1] = Iext(kwargs, tList[_i+1])
    return soln, Ilist