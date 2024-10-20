#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:19:27 2024

@author: reinierramos
"""

import numpy as np

from numpy import typing as npt
from numpy import random as nrand
from .odesolvers import (lsoda, euler, rk4)

rng = nrand.default_rng(17)
solvers = {'lsoda': lsoda, 'euler': euler, 'rk4':rk4}

class SolverError(Exception):
    """
    Throws the error if solver is set to 'lsoda' but the system
    contains 'noisy' or 'coupled'.
    """
    def __init__(self, system, solver
                , msg='LSODA is incompatible to noisy and coupled systems.'):
        self.system=system
        self.solver=solver
        super().__init__(msg)

def makeTimeList(ti:float=0, tf:float=100, 
                 dt:float=0.025) -> npt.NDArray[float]:
    """
    Generates an (d,) array of t values from ti to tf with steps dt, 
    where d = (tf-ti)/dt.

    """
    return np.arange(ti, tf, dt)

def solveWC(system='single', solver='euler', E0=0, I0=0, I0_e=0, I0_i=0,
            Is_e=0, Is_i=0, fs=0, ti=0, tf=100, dt=0.025, **kwargs):
    """
    Solves the Wilson-Cowan (WC) system using solver. 

    Parameters
    ----------
    system : str, default is 'single'
        Type of WC system to solve.
        Accepted values are: 'single', 'noisy'.
    solver : str, default is 'euler'
        Method of solving ODEs.
        Accepted values are: 'lsoda', 'euler', 'rk4'.
    E0 : float, default is 0
        Initial density of excitatory neurons.
        Must be between [0,1].
    I0 : float, default is 0
        Initial density of inhibitory neurons.
        Must be between [0,1].
    I0_e : float, default is 0
        Amplitude in uA/cm^2, of the constant current for excitatory neurons.
    I0_i : float, default is 0
        Amplitude in uA/cm^2, of the constant current for inhibitory neurons.
    Is_e : float, default is 0
        Amplitude in uA/cm^2, of the sinusoidal current for excitatory neurons.
    Is_i : float, default is 0
        Amplitude in uA/cm^2, of the sinusoidal current for inhibitory neurons.
    fs : float, default is 0
        Frequency in Hz, of the sinusoidal input for both excitatory 
        and inhibitory neurons.
        Must be nonzero when `Is_e`, `Is_i`, or both are nonzero.
    ti : float, default is 0
        Initial time in ms, for stimulus duration.
    tf : float, default is 100
        Final time in ms, for stimulus duration.
    dt : float, default is 0.025
        Time step in ms. 
        The stimulus duration is obtained as `np.arange(ti,tf,dt)`.
    **kwargs : dict
        Extra args for other parameters defining the input current such as,
        In : float
            Amplitude in uA/cm^2, of noisy input.
            Must be passed if system is 'noisy'.

    Raises
    ------
    SolverError
        If solver is 'LSODA' but the system is either 'noisy'.
    
    Returns
    -------
    soln : (2, d) or (2, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (E, I) for all t in tList.
    I_elist : (d,) or (P, d) array
        Total input current for excitatory neurons.
    I_ilist : (d,) or (P, d) array
        Total input current for inhibitory neurons.
    tList : (d,) array
        Time points for which WC is evaluated.
    
    """
    if solver=='lsoda' and system!='single': 
        raise SolverError(system, solver)
    tList = makeTimeList(ti, tf, dt)
        
    if 'noisy' in system:
        noise = rng.random(len(tList)) - 0.5
        kwargs.update({'noise':noise})
    kwargs.update({'system':system, 'E0':E0, 'I0':I0, 'I0_e':I0_e, 
                   'I0_i':I0_i, 'Is_e':Is_e, 'Is_i':Is_i, 'fs':fs})
    solver_ = solvers.get(solver)
    soln, I_elist, I_ilist = solver_(tList, kwargs)
    return soln.T, I_elist, I_ilist, tList