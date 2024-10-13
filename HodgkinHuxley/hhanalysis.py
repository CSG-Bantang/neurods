#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:42:27 2024

@author: reinierramos
"""

import numpy as np
import networkx as nx
from .hhsolvers import lsoda, euler, rk4
from numpy import random as nrand
from .sschannels import (m_inf, h_inf, n_inf, tau_h, tau_m, tau_n)

rng = nrand.default_rng(17)

solvers = {'lsoda': lsoda, 'euler': euler, 'rk4':rk4}

class SolverError(Exception):
    def __init__(self, system, solver
                , msg='LSODA is incompatible to noisy and coupled systems.'):
        self.system=system
        self.solver=solver
        super().__init__(msg)

def makeTimeList(ti=0, tf=100, dt=0.025):   return np.arange(ti, tf, dt)

def solveHH(system='single', solver='euler', Vrest=0,
            I0=0, Is=0, fs=0, ti=0, tf=100, dt=0.025, **kwargs):
    """
    Solves the Hodgkin-Huxley (HH) `system` using `solver`. 

    Parameters
    ----------
    system : str, default is 'single'
        Type of HH system to solve.
        Accepted values are: 'single', 'noisy', 'coupled', 'noisy coupled'.
    solver : str, default is 'euler'
        Method of solving ODEs
        Accepted values are: 'lsoda', 'euler', 'rk4'.
    Vrest : float, default is 0
        Resting voltage.
    I0 : float, default is 0
        Amplitude, in uA/cm^2, of the constant or bias current.
        If no dynamics is observed, then provide a nonzero value.
    Is : float, default is 0
        Amplitude, in uA/cm^2, of the sine input.
    fs : float, default is 0
        Frequency, in Hz, of the sine input.
        Must be nonzero when `Is` is nonzero.
    ti : float, default is 0
        Initial time, in ms, for stimulus duration.
    tf : float, default is 100
        Final time, in ms, for stimulus duration.
    dt : float, default is 0.025
        Timestep size, in ms.
        The stimulus duration is obtained as `np.arange(ti,tf,dt)`.
    **kwargs : dict
        Container for parameters valid for each type of stimulus input.

    Raises
    ------
    SolverError
        If `solver` is 'LSODA' but the `system` is either 'noisy', 'coupled', 
        or 'noisy coupled.'

    Returns
    -------
    soln : 2D or 3D ndarray
        Values of V, m, h, n for all `t` in `tList`.
    tList : 1D ndarray
        Time points for which HH is evaluated.

    Valid keywords in `**kwargs`:
        In : float
            Amplitude, in uA/cm^2, of the noisy input.
        L : int
            Lattice size.
        g : float
            Uniform coupling strength of each neuron to its neighbors in the lattice.

    """
    if solver=='lsoda' and system!='single':  raise SolverError(system, solver)
    
    tList = makeTimeList(ti, tf, dt)
    
    if 'coupled' in system:
        L = kwargs.get('L')
        population = L*L
        G = nx.grid_2d_graph(L,L)  
        adjMat = nx.to_numpy_array(G)
        adjMat = np.triu(adjMat, k=0)
        kwargs.update({'aij':adjMat, 'pop':population})
        
    if 'noisy' in system:
        noise = rng.random(len(tList)) - 0.5
        kwargs.update({'noise':noise})
    kwargs.update({'Vrest':Vrest, 'I0':I0, 'Is':Is, 'fs':fs, 'dt':dt, 'system':system})
    solver_ = solvers.get(solver)
    soln = solver_(tList, kwargs).T
    return soln, tList

def asymptoticChannels(Vmin=-100, Vmax=100, steps=400):
    Vspace = np.linspace(-100, 100, 400)
    return (Vspace, m_inf(Vspace), h_inf(Vspace), n_inf(Vspace))

def timeConstants(Vmin=-100, Vmax=100, steps=400):
    Vspace = np.linspace(-100, 100, 400)
    return (Vspace, tau_m(Vspace), tau_h(Vspace), tau_n(Vspace))

def firingRate(V, t, Vthresh=19):
    return (np.count_nonzero(np.diff(1*(V > Vthresh))==1)-1)/(t[-1]-t[0])

    
