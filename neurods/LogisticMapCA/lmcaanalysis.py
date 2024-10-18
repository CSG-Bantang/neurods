#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:06:36 2024

@author: reinierramos
"""

import numpy as np
import itertools as itools
from .lmcautils import updateGrid

from numpy import random as nrand
rng = nrand.default_rng(17)

def logisticEquation(r:float=1.0, xt:float|list=0.5) -> float|list:
    """
    Calculates the next value x(t+1) according to a logistic growth rate r
    given the current value xt.

    """
    return r*xt*(1-xt)

def solveLM(r=1.0, x0=0.5, ti=0, tf=50, dt=1):
    """
    Solves logistic map (LM) equation given a duration.

    Parameters
    ----------
    r : float, default is 1.0
        Logistic growth rate.
        Must be between 0 and 4.
    x0 : float, default is 0.5
        Normalized initial state of the LM system.
        Must be between 0 and 1.
    ti : float, default is 0
        Initial time, in arb. time unit.
    tf : float, default is 50
        Final time, in arb. time unit.
    dt : float, default is 1
        Time step, in arb. time unit.

    Returns
    -------
    xList : (d,) array
        where d = (tf-ti)/dt
        Values of x(t).
    tList : (d,) array
        where d = (tf-ti)/dt
        Time points for which LM is evaluated.

    """
    tList = np.arange(ti,tf,dt)
    xList = np.zeros(len(tList))
    xList[0] = x0
    for _i in range(len(tList)-1):
        xList[_i+1] = logisticEquation(r, xList[_i])
    return xList, tList


def logisticReturnMap(r:float=1.0):
    """
    Input-output map for the logistic equation 
    x[t+1] = r * (1-x[t]) * x[t].

    Parameters
    ----------
    r : float, default is 1.0
        Logistic growth rate.
        Must be between 0 and 4.

    Returns
    -------
    x : (numpoints,) array
        Input states x[t].
    y : (numpoints,) array
        Output states x[t+1].

    """
    numpoints = 300
    x = np.linspace(0, 1, numpoints)
    y = logisticEquation(r, x)
    return x, y

def solveLMCA(system=1, duration=50, L=50, r=4, init='uniform', **kwargs):
    """
    Solves the spatiotemporal snapshots of a Logistic CA with lattice size L.
    The system specifies the neighborhood and lattice boundary conditions, 
    while the CA is initialized with random distribution specified by init.

    Parameters
    ----------
    system : int, default is 1
        Specifies neighborhood and lattice boundary conditions.
        1: toroidal,  outer-totalistic Moore
        2: toroidal,  inner-totalistic Moore
        3: toroidal,  outer-totalistic von Neumann
        4: toroidal,  outer-totalistic von Neumann
        5: spherical, outer-totalistic Moore
        6: spherical, inner-totalistic Moore
        7: spherical, outer-totalistic von Neumann
        8: spherical, inner-totalistic von Neumann
    duration : int, default is 50
        Number of timesteps to solve Logistic CA.
        Must be nonzero.
    L : int, default is 50
        Lattice size for the LCA.
        Must be greater than 0.
    r : float, default is 4
        Logistic growth rate.
        Must be between [0, 4].
    init : str, default is 'uniform'
        Random distribution to initialize Logistic CA.
        Accepted values: 'uniform' and 'beta'.
        If 'beta', additional `kwargs must be provided, 
        either ``(a=, b=)`` or ``(mu=, nu=)``.
    **kwargs : dict
        Additional arguments needed to specify shape parameters of beta 
        distribution. If mu and nu are specified, then the parameters a
        and b are automatically computed. Either choose ``(a=, b=)`` or 
        ``(mu=, nu=)``.
        a: float, must be nonzero, a=mu*nu
        b: float, must be nonzero, b=(1-mu)*nu
        mu: float, mean, Must be in (0,1).
        nu: float, precision of the mean, "sample size" in Bayes theorem.

    Returns
    -------
    soln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of Logistic CA.

    """
    if init=='beta' and kwargs.get('a') and kwargs.get('b'):
        a, b = kwargs.get('a'), kwargs.get('b')
        grid = rng.beta(a, b, size=(L, L), dtype=np.float32)
    if init=='beta' and kwargs.get('mu') and kwargs.get('nu'):
        mu, nu = kwargs.get('mu'), kwargs.get('nu')
        a, b = mu*nu, (1-mu)*nu
        grid = rng.beta(a, b, size=(L, L), dtype=np.float32)
    if init=='uniform':
        grid = rng.random(size=(L,L), dtype=np.float32)
    grid_coords = list(itools.product(range(L), repeat=2))
    
    soln = np.zeros((duration+1, L,L), dtype=np.float32)
    soln[0,:,:] = grid
    
    propsCA = {'system': system, 'L': L, 'r':r}
    
    for t in range(duration):
        grid = updateGrid(L, grid, grid_coords, propsCA)
        soln[t+1,:,:] = grid
    return soln