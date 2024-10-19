#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 00:31:50 2024

@author: reinierramos
"""

import numpy as np
from .ecautils import (updateLattice)

def solveECA(rule=30, init=1, duration=150, L=150, dens1=0.5):
    """
    Solves the spatiotemporal snapshots of a Wolfram's Elementary CA with a 
    lattice length L for a given duration and rule. The init specifies how
    the lattice is initialized.
    
    Parameters
    ----------
    rule : int, default is 30
        Converted into an 8-digit binary and is mapped 
        into each possible states of neighborhood.
        Must be from [0,255].
    init : int, default is 1
        Specifies how to initialize the lattice.
        If 1, then a single "Active" state is initialized in the middle.
        If 0, then the lattice is initialized randomly with density 
        of "Active" states given by dens1.
    duration : int, default is 150
        Number of timesteps to solve Wolfram's ECA.
        Must be nonzero.
    L : int, default is 150
        Lattice length for the Wolfram's ECA.
        A 1D ring boundary condition is generated.
        Must be greater than 2.
    dens1 : float, default is 0.5
        Initial density of "Active":"1" cells in the CA.
        Must be between [0,1].

    Returns
    -------
    soln : (duration, L) array
        Snapshots of the spatiotemporal dynamics of Wolfram's ECA.
    
    See Also
    --------
    ecaRules(rule, nbc) for an example of rule.

    """
    soln = np.zeros((duration+1, L), dtype=int)
    mid = int(L/2)
    if init:
        lattice = np.zeros(L, dtype=int)
        lattice[mid] = 1
    else:
        dens0 = 1 - dens1
        lattice = np.random.choice([0,1], size=L, p=[dens0, dens1])
    soln[0] = lattice
    
    for t in range(duration):
        lattice = updateLattice(rule, L, lattice, mid)
        soln[t+1,:] = lattice
    return soln