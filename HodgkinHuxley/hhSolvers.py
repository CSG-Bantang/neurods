#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:45:44 2024

@author: reinierramos
"""

import numpy as np
from .sschannels import (m_inf, h_inf, n_inf)
from .hhodes import odes
from scipy.integrate import odeint


def lsoda(tList, hhparams):
    """
    Solve the ODEs using LSODA (Livermore Solver for Ordinary Differential
    Equations) via the implementation of `scipy.integrate.odeint`).

    Parameters
    ----------
    tList : 1D array
        List of time values to solve the ODE.
    hhparams : dict
        Container for parameters valid for each type of stimulus input.

    Returns
    -------
    soln : 2D ndarray
        Values of V, m, h, n for al `t` in `tList`.

    Valid keywords in `hhparams`:
        Vrest : float
            Resting voltage.
        system : str
            Type of HH system ('single', 'noisy', 'coupled', 'noisy coupled').
        dt : float
            Timestep size, in ms.
        I0 : float
            Amplitude, in uA/cm^2, of the constant or bias current.
        Is : float
            Amplitude, in uA/cm^2, of the sine input.
        fs : float
            Frequency, in Hz, of the sine input.

    """
    Vrest = hhparams.get('Vrest')
    guess = [Vrest, m_inf(Vrest), h_inf(Vrest), n_inf(Vrest)]
    soln   = odeint(odes, guess, tList, args=(hhparams,))
    return soln

def euler(tList, hhparams):
    """
    Solve the ODEs using the forward Euler method.

    Parameters
    ----------
    tList : 1D array
        List of time values to solve the ODE.
    hhparams : dict
        Container for parameters valid for each type of stimulus input.

    Returns
    -------
    soln : 2D ndarray
        Values of V, m, h, n for al `t` in `tList`.
        
    Valid keywords in `params_`:
        Vrest : float
            Resting voltage.
        system : str
            Type of HH system ('single', 'noisy', 'coupled', 'noisy coupled').
        dt : float
            Timestep size, in ms.
        I0 : float
            Amplitude, in uA/cm^2, of the constant or bias current.
        Is : float
            Amplitude, in uA/cm^2, of the sine input.
        fs : float
            Frequency, in Hz, of the sine input.
        In : float
            Amplitude, in uA/cm^2, of the noisy input.
        noise : 1D ndarray
            List of generated random numbers from a uniform distribution [-0.5, 0.5].
        L : int
            Lattice size.
        pop : int
            Total number of neurons in the square lattice of size `L`.
        aij : 2D ndarray
            Adjacency matrix of the square lattice.
        g : float
            Uniform coupling strength of each neuron to its neighbors in the lattice.

    """
    dt = hhparams.get('dt')
    Vrest = hhparams.get('Vrest')
    guess = np.array([Vrest,m_inf(Vrest), h_inf(Vrest), n_inf(Vrest)])
    if 'coupled' in hhparams.get('system'):
        soln = np.zeros([hhparams.get('pop'), len(tList), 4])
        soln[:,0,:] = guess
        guess = soln[:,0,:]
    else:
        soln    = np.zeros([len(tList), 4])
        soln[0] = guess
    for _i in range(len(tList)-1):
        if 'noisy' in hhparams.get('system'):
            noise_ = hhparams.get('noise')[_i]
            hhparams.update({'noise_t': noise_})
        if 'coupled' in hhparams.get('system'):
            Vi = np.tile(soln[:,_i,0], (hhparams.get('pop'),1))
            Vj = Vi.T
            Vij = Vi-Vj
            hhparams.update({'Vij':Vij})
        next_ = odes(guess, tList[_i+1], hhparams).T
        guess += next_*dt
        if 'coupled' in hhparams.get('system'): soln[:,_i+1, :] = guess
        else: soln[_i+1] = guess
    return soln

def rk4(tList, hhparams):
    """
    Solve the ODEs using the 4th-order Runge-Kutta method.

    Parameters
    ----------
    tList : 1D array
        List of time values to solve the ODE.
    hhparams : dict
        Container for parameters valid for each type of stimulus input.

    Returns
    -------
    soln : 2D ndarray
        Values of V, m, h, n for al `t` in `tList`.
        
    Valid keywords in `params_`:
        Vrest : float
            Resting voltage.
        system : str
            Type of HH system ('single', 'noisy', 'coupled', 'noisy coupled').
        dt : float
            Timestep size, in ms.
        I0 : float
            Amplitude, in uA/cm^2, of the constant or bias current.
        Is : float
            Amplitude, in uA/cm^2, of the sine input.
        fs : float
            Frequency, in Hz, of the sine input.
        In : float
            Amplitude, in uA/cm^2, of the noisy input.
        noise : 1D ndarray
            List of generated random numbers from a uniform distribution [-0.5, 0.5].
        L : int
            Lattice size.
        pop : int
            Total number of neurons in the square lattice of size `L`.
        aij : 2D ndarray
            Adjacency matrix of the square lattice.
        g : float
            Uniform coupling strength of each neuron to its neighbors in the lattice.

    """
    dt = hhparams.get('dt')
    Vrest = hhparams.get('Vrest')
    guess = np.array([Vrest, m_inf(Vrest), h_inf(Vrest), n_inf(Vrest)])
    if 'coupled' in hhparams.get('system'):
        soln = np.zeros([hhparams.get('pop'), len(tList), 4])
        soln[:,0,:] = guess
        guess = soln[:,0,:]
    else:
        soln    = np.zeros([len(tList), 4])
        soln[0] = guess
    for _i in range(len(tList)-1):
        if 'noisy' in hhparams.get('system'):
            noise_ = hhparams.get('noise')[_i]
            hhparams.update({'noise_t': noise_})
        if 'coupled' in hhparams.get('system'):
            Vi = np.tile(soln[:,_i,0], (hhparams.get('pop'),1))
            Vj = Vi.T
            Vij = Vi-Vj
            hhparams.update({'Vij':Vij})
        k1 = dt * odes(guess,        tList[_i+1],        hhparams).T
        k2 = dt * odes(guess+0.5*k1, tList[_i+1]+0.5*dt, hhparams).T
        k3 = dt * odes(guess+0.5*k2, tList[_i+1]+0.5*dt, hhparams).T
        k4 = dt * odes(guess+k3,     tList[_i+1]+dt,     hhparams).T
        guess += (k1 + 2*(k2+k3) + k4)/6
        if 'coupled' in hhparams.get('system'): soln[:,_i+1, :] = guess
        else: soln[_i+1] = guess
    return soln
