#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 20:46:44 2024

@author: reinierramos
"""

import numpy as np
import networkx as nx
from numpy import typing as npt
from numpy import random as nrand

from .odesolvers import (lsoda, euler, rk4)
from .steadystate import (inf_m, inf_h, inf_n, tau_m, tau_h, tau_n)

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

def makeTimeList(ti:float=0, tf:float=100, dt:float=0.025) -> npt.NDArray[float]:
    """
    Generates an (d,) array of t values from ti to tf with steps dt, 
    where d = (tf-ti)/dt.

    """
    return np.arange(ti, tf, dt)

def solveHH(system='single', solver='euler', Vrest=0,
            I0=0, Is=0, fs=0, ti=0, tf=100, dt=0.025, **kwargs):
    """
    Solves the Hodgkin-Huxley (HH) system using solver. 

    Parameters
    ----------
    system : str, default is 'single'
        Type of HH system to solve.
        Accepted values are: 'single', 'noisy', 'coupled', 'noisy coupled'.
    solver : str, default is 'euler'
        Method of solving ODEs.
        Accepted values are: 'lsoda', 'euler', 'rk4'.
    Vrest : float, default is 0
        Resting voltage.
    I0 : float, default is 0
        Amplitude in uA/cm^2, of the constant or bias current.
        If no dynamics is observed, then provide a nonzero value.
    Is : float, default is 0
        Amplitude in uA/cm^2, of the sine input.
    fs : float, default is 0
        Frequency in Hz, of the sine input.
        Must be nonzero when `Is` is nonzero.
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
            Must be passed if system is 'noisy' or 'noisy coupled'.
        L : int
            Lattice size of the HH network.
            Must be passed if system is 'coupled' or 'noisy coupled'.
        g : float
            Uniform coupling strength between neighboring neurons in 
            the lattice. Must be passed if system is 'coupled' or 
            'noisy coupled'.

    Raises
    ------
    SolverError
        If solver is 'LSODA' but the system is either 'noisy', 'coupled', 
        or 'noisy coupled.'
    
    Returns
    -------
    soln : (4, d) or (4, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (Phi, m, h, n) for all t in tList.
    Ilist : (d,) or (P, d) array
        Total input current for all t in tlist.
    tList : (d,) array
        Time points for which HH is evaluated.
    
    """
    if solver=='lsoda' and system!='single': 
        raise SolverError(system, solver)
    tList = makeTimeList(ti, tf, dt)
    if 'coupled' in system:
        L = kwargs.get('L')
        G = nx.grid_2d_graph(L,L)  
        adjMat = nx.to_numpy_array(G)
        adjMat = np.triu(adjMat, k=0)
        kwargs.update({'aij':adjMat})
        
    if 'noisy' in system:
        noise = rng.random(len(tList)) - 0.5
        kwargs.update({'noise':noise})
    kwargs.update({'system':system, 'Vrest':Vrest, 'I0':I0, 'Is':Is, 'fs':fs})
    solver_ = solvers.get(solver)
    soln, Ilist = solver_(tList, kwargs)
    return soln.T, Ilist, tList

def channelAsymptotes(Vmin=-100, Vmax=100, numpoints=400):
    """
    Solves the asymptotic values of (m, h, n) as t approaches infinity, given
    a list of resting voltage from Vmin to Vmax.

    Parameters
    ----------
    Vmin : float, default is -100
        Minimum resting voltage.
    Vmax : float, default is 100
        Maximum resting voltage.
    numpoints : int, default is 400
        Number of voltage values linearly spaced from Vmin to Vmax.

    Returns
    -------
    Phispace : (numpoints,) array
        Range of voltage values from Vmin to Vmax.
    m_inf : (numpoints,) array
        Corresponding asymptotic values of m channel.
    h_inf : (numpoints,) array
        Corresponding asymptotic values of h channel.
    n_inf : (numpoints,) array
        Corresponding asymptotic values of n channel.

    """
    Phispace = np.linspace(Vmin, Vmax, numpoints)
    return (Phispace, inf_m(Phispace), inf_h(Phispace), inf_n(Phispace))

def timeConstants(Vmin=-100, Vmax=100, numpoints=400):
    """
    Solves the time constants of (m, h, n) channels at steady-state given
    a list of resting voltage from Vmin to Vmax.

    Parameters
    ----------
    Vmin : float, default is -100
        Minimum resting voltage.
    Vmax : float, default is 100
        Maximum resting voltage.
    numpoints : int, default is 400
        Number of voltage values linearly spaced from Vmin to Vmax.

    Returns
    -------
    Phispace : (numpoints,) array
        Range of voltage values from Vmin to Vmax.
    m_tau : (numpoints,) array
        Corresponding time constant values of m channel.
    h_tau : (numpoints,) array
        Corresponding time constant values of h channel.
    n_tau : (numpoints,) array
        Corresponding time constant values of n channel.

    """
    Phispace = np.linspace(Vmin, Vmax, numpoints)
    return (Phispace, tau_m(Phispace), tau_h(Phispace), tau_n(Phispace))

def firingRate(V, t, Vthresh=19):
    """
    Solves the firing rate or interspike interval (ISI) of an HH neuron.
    A spike is counted once the voltage V increases and crosses the 
    threshold voltage Vthresh.

    Parameters
    ----------
    V : (d,) array
        where d = (tf-ti)/dt.
        Voltage values of the HH neuron over time
    t : (d,) array
        where d = (tf-ti)/dt
        Time values for which V is recorded.
    Vthresh : float, default is 19
        Threshold voltage to generate a spike.

    Returns
    -------
    ISI : float
        Firing rate of a neuron for a given duration and threshold voltage.

    """
    return (np.count_nonzero(np.diff(1*(V > Vthresh))==1)-1) / (t[-1]-t[0])
    