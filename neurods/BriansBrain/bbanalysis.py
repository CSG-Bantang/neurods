#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:14:11 2024

@author: reinierramos
"""

import numpy as np
import itertools as itools

from .bbutils import updateGrid

from numpy import random as nrand
rng = nrand.default_rng(17)

def solveBB(system=1, duration=30, L=50, dq=1/3, df=1/3, k=1,
            tRefrac=1, Lambda=2, firingRule='='):
    """
    Solves the spatiotemporal snapshots of a Brian's Brain (BB) CA with a 
    lattice size L for a given duration. The default values for other args
    corresponds to the original Brian's Brian rules and conditions.

    Parameters
    ----------
    system : int, default is 1
        Neighborhood boundary condition.
        1: outer-totalistic Moore
        2: inner-totalistic Moore
        3: outer-totalistic von Neumann
        4: outer-totalistic von Neumann
    duration : int, default is 30
        Number of timesteps to solve BB CA.
        Must be nonzero.
    L : int, default is 50
        lattice size for the BB CA. 
        A square lattice with toroidal boundary conditions is generated.
        Must be greater than 0.
    dq : float, default is 1/3
        Initial density of "Q" cells in the CA.
        Must be between [0,1].
        `Note: total must be dq+df+dr=1.`
    df : float, default is 1/3
        Initial density of "F" cells in the CA.
        Must be between [0,1].
        `Note: total must be dq+df+dr=1`.
    k : int, default is 1
        Radius of the neighborhood specified by system.
        Must be k < L//2.
    tRefrac : int, default is 1
        Refractory period. 
        Number of timesteps that a neuron will stay "R".
        Must be nonzero.
    Lambda : int, default is 2
        Firing threshold.
        See `firingRule` for the interpretation.
    firingRule : str, default is '='
        Firing condition that sets inequality for `Lambda`.
        Interpreted as: "Q"->"F" if num_F_neighbors("Q") = Lambda.
        Accepted values: '=', '>=', '<=', '>', '<'.

    Returns
    -------
    soln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of BB CA.

    """
    dr = 1 - (dq+df)
    
    grid = rng.choice([0,1,2], size=(L,L), p=(dq,df,dr)).astype(np.int32)
    grid_coords = list(itools.product(range(L), repeat=2))
    
    gridRefrac = np.zeros((L,L), dtype=int)
    gridRefrac[grid==2] = 1
    
    soln = np.zeros((duration+1, L,L), dtype=np.int32)
    soln[0,:,:] = grid
    
    propsCA = {'system':system,     'k':k,  'L': L,
              'lambda':Lambda,          'firingRule': firingRule,
              'timeRefrac':tRefrac,     'gridRefrac':gridRefrac}
    for t in range(duration):
        grid = updateGrid(L, grid, grid_coords, propsCA)
        gridRefrac[grid==2] += 1
        gridRefrac[grid==0] = 0
        gridRefrac[grid==1] = 0
        propsCA.update({'gridRefrac':gridRefrac})
        soln[t+1,:,:] = grid
    return soln