#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:42:25 2024

@author: reinierramos
"""

import numpy as np
import itertools as itools
from numpy import random as nrand
from .ncalutils import (applyDefect, updateGrid)

rng = nrand.default_rng(17)

def solveNCAL(system=1, duration=30, layers=2, L=50, a0=0.0, a1=0.8, a2=0.9, 
              nl=2.0, init='uniform', defect=0., **kwargs):
    """
    Solves the spatiotemporal snapshots of a Neuronal CA with layers of 
    lattice size L. The system specifies the neighborhood and lattice boundary 
    conditions, while the CA is initialized with random distribution specified 
    by init.

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
    duration : int, default is 30
        Number of timesteps to solve Layered Neuronal CA.
        Must be nonzero.
    layers : int, default is 2
        Number of layers of the Neuronal CA.
    L : int, default is 50
        Lattice size for the LCA.
        Must be greater than 0.
    a0 : float, default is 0.0
        Minimum input threshold
        Must be between [0,1].
    a1 : float, default is 0.8
        Maximum input threshold
        Must be between [0,1].
    a2 : float, default is 0.9
        Maximum output threshold
        Must be between [0,1].
    nl : float, default is 1.0
        Nonlinearity of activation profile.
        Must be greater than 0.
    init : str, default is 'uniform'
        Random distribution to initialize Layered Neuronal CA.
        Accepted values: 'uniform' and 'beta'.
        If 'beta', additional `kwargs must be provided, 
        either ``(a=, b=)`` or ``(mu=, nu=)``.
    defect : float, default is 0
        Density of spike-defective neurons.
        Must be between [0,1].
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
    soln : (duration, layer, L, L) array
        Snapshots of the spatiotemporal dynamics of Layaered Neuronal CA.
    defectIndices : optional, list with length int(defect*layers*L*L)
        Returns list of tuples indicating the CA indices of spike-defective 
        neurons if defect is nonzero.

    """
    if init=='beta' and kwargs.get('a') and kwargs.get('b'):
        a, b = kwargs.get('a'), kwargs.get('b')
        grid = rng.beta(a, b, size=(layers, L, L), dtype=np.float32)
    if init=='beta' and kwargs.get('mu') and kwargs.get('nu'):
        mu, nu = kwargs.get('mu'), kwargs.get('nu')
        a, b = mu*nu, (1-mu)*nu
        grid = rng.beta(a, b, size=(layers, L, L), dtype=np.float32)
    if init=='uniform':
        grid = rng.random(size=(layers, L,L), dtype=np.float32)
    grid_coords = list(itools.product(range(layers), range(L), range(L)))
    
    if defect:
        rng.shuffle(defectIndex := grid_coords.copy())
        grid = applyDefect(grid, defectIndex[:int(defect*layers*L*L)])
    
    soln = np.zeros((duration+1,layers,L,L), dtype=np.float32)
    soln[0,:,:,:] = grid
    
    propsCA = {'system':system, 'L':L, 'layers':layers, 
               'a0':a0, 'a1':a1, 'a2':a2, 'nl':nl}
    
    for t in range(duration):
        grid = updateGrid(L, grid, grid_coords, propsCA)
        if defect: grid = applyDefect(grid, 
                                      defectIndex[:int(defect*layers*L*L)])
        soln[t+1,:,:,:] = grid
    if defect:
        return soln, defectIndex[:int(defect*layers*L*L)]
    return soln

def ncaReturnMap(a0:float, a1:float, a2:float, nl:float) -> tuple[list[float]]:
    """
    Input-output map for the activation equation 
    a[t+1] = a2(1 - (1 - (a[t]-a0)/(a1-a0))^b) for a1 > a[t] > a0,
    zero otherwise.

    Parameters
    ----------
    a0 : float
        Minimum input threshold
        Must be between [0,1].
    a1 : float
        Maximum input threshold
        Must be between [0,1].
    a2 : float
        Maximum output threshold
        Must be between [0,1].
    nl : float
        Nonlinearity of activation profile.
        Must be greater than 0.

    Returns
    -------
    x : (numpoints,) array
        Input states a_in = a[t].
    y : (numpoints,) array
        Output states a_out = a[t+1].
    
    """
    numpoints = 300
    x = np.linspace(0, 1, numpoints)
    y = np.array([activationEquation(a, a0, a1, a2, nl) for a in x])
    return x, y

def activationEquation(a_in:float, a0:float, a1:float, a2:float, 
                       nl:float) -> float:
    """
    Solves for the next value a_out given the input a_in 
    using the activation equation.

    """
    if a_in == 0 and a0 > a1: 
        return a2
    elif min(a0,a1) == 1 and a_in == min(a0,a1):
        return a2
    elif min(a0,a1) == 1 and a_in != min(a0,a1):
        return 0.
    elif a_in == max(a0,a1) and a0 > a1:
        return 0
    elif a_in == max(a0,a1) and a1 > a0:
        return a2
    elif max(a0,a1) >= a_in >= min(a0,a1):
        return a2 * (1-np.exp((-nl) * (-np.log(1-(a_in-a0)/(a1-a0))) ))
    else:
        return 0