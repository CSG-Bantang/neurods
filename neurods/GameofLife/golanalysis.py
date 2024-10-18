#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:26:30 2024

@author: reinierramos
"""

import numpy as np
import itertools as itools
from numpy import random as nrand
from .golsystems import initializeGOL
from .golutils import updateGrid

rng = nrand.default_rng(17)

def solveGOL(system=0, L=50, p=0.5, duration=30):
    """
    Solves the spatiotemporal snapshot of a Game of Life (GOL) CA.
    If system is 0, then the CA is initalized in a lattice of size L 
    with uniform random distribution of states with densities
    "alive":p and "dead":1-p.
    
    If system is not 0, then the CA is initialized with predefined life-forms
    (see README.md for more information).
    
    Parameters
    ----------
    system : int, default is 0
        Determines initial state of the CA.
        If 0, then a random state is initialized.
        Accepted values are 0 to 17.
    L : int, default is 50
        Lattice size for the GOL CA.
        This will be ignored if system is not 0.
    p : float, default is 0.5
        Initial density of "alive" cells in the CA.
        Must be between [0,1].
        This will be ignored if system is not 0.
    duration : int, default is 30
        Number of timesteps to solve GOL CA.

    See Also
    --------
    initializeGOL(system) -> Predefined patterns
    
    Returns
    -------
    soln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of GOL CA.

    """
    if not system:
        grid = rng.choice([0,1], size=(L,L), replace=True, 
                          p=(1-p,p)).astype(np.int32)
    else:
        _ini = initializeGOL(system)
        L = int(np.sqrt(len(_ini)))
        grid = _ini.reshape((L,L))
    
    grid_coords = list(itools.product(range(L), repeat=2))
    soln = np.zeros((duration+1, L,L))
    soln[0,:,:] = grid
    
    for t in range(duration):
        grid = updateGrid(L, grid, grid_coords)
        soln[t+1,:,:] = grid
    return soln