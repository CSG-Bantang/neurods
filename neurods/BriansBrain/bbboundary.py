#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:29:51 2024

@author: reinierramos
"""

import numpy as np
# import numba as nb  ### FixMe: numba incompatible with np.roll with axis arg.

def getNeighbors(propsCA:dict, grid:list[int], j:int, i:int) -> list[int]:
    """
    Returns neighbors of the cell grid[j,i] given the following 
    keys from propsCA:
        'system' : int, default is 1
            Neighborhood boundary condition.
            1: outer-totalistic Moore
            2: inner-totalistic Moore
            3: outer-totalistic von Neumann
            4: outer-totalistic von Neumann
        'L': int, lattice size
            L > 0.
        'k' : int, default is 1
            Radius of the neighborhood specified by system.
            Must be k < L//2.
        
    """
    L = propsCA.get('L')
    k = propsCA.get('k')
    system = propsCA.get('system')
    func = nbc.get(system)
    return func(L,grid,j,i,k)

# @nb.njit(nb.int32[:](nb.int32, nb.int32[:,:], nb.int32, nb.int32, nb.int32))
def Moore_inner(L, grid, j, i, k):
    rolled = np.roll(grid, shift=(int(L/2)-j, int(L/2)-i), axis=(0,1))
    j = int(L//2)
    i = int(L//2)
    neighbors = np.zeros(4*k+4*k*k+1)
    neighbors[   :k]   = rolled[j-k:j,i]
    neighbors[  k:2*k] = rolled[j+1:j+1+k,i]
    neighbors[2*k:3*k] = rolled[j,i-k:i]
    neighbors[3*k:4*k] = rolled[j,i+1:i+1+k]
    neighbors[4*k      :4*k+1*k*k] = rolled[j-k:j,i-k:i].ravel()
    neighbors[4*k+1*k*k:4*k+2*k*k] = rolled[j-k:j,i+1:i+1+k].ravel()
    neighbors[4*k+2*k*k:4*k+3*k*k] = rolled[j+1:j+k+1,i-k:i].ravel()
    neighbors[4*k+3*k*k:4*k+4*k*k] = rolled[j+1:j+k+1,i+1:i+1+k].ravel()
    neighbors[-1] = rolled[j,i]
    return neighbors

# @nb.njit(nb.int32[:](nb.int32, nb.int32[:,:], nb.int32, nb.int32, nb.int32))
def Moore_outer(L, grid, j, i, k):
    rolled = np.roll(grid, shift=(int(L/2)-j, int(L/2)-i), axis=(0,1))
    j = int(L//2)
    i = int(L//2)
    neighbors = np.zeros(4*k+4*k*k)
    neighbors[   :k]   = rolled[j-k:j,i]
    neighbors[  k:2*k] = rolled[j+1:j+1+k,i]
    neighbors[2*k:3*k] = rolled[j,i-k:i]
    neighbors[3*k:4*k] = rolled[j,i+1:i+1+k]
    neighbors[4*k      :4*k+1*k*k] = rolled[j-k:j,i-k:i].ravel()
    neighbors[4*k+1*k*k:4*k+2*k*k] = rolled[j-k:j,i+1:i+1+k].ravel()
    neighbors[4*k+2*k*k:4*k+3*k*k] = rolled[j+1:j+k+1,i-k:i].ravel()
    neighbors[4*k+3*k*k:4*k+4*k*k] = rolled[j+1:j+k+1,i+1:i+1+k].ravel()
    return neighbors

# @nb.njit(nb.int32[:](nb.int32, nb.int32[:,:], nb.int32, nb.int32, nb.int32))
def vonNeumann_inner(L, grid, j, i, k):
    rolled = np.roll(grid, shift=(int(L/2)-j, int(L/2)-i), axis=(0,1))
    j = int(L//2)
    i = int(L//2)
    neighbors = np.zeros(4*k+4*k*k+1)
    neighbors[   :k]   = rolled[j-k:j,i]
    neighbors[  k:2*k] = rolled[j+1:j+1+k,i]
    neighbors[2*k:3*k] = rolled[j,i-k:i]
    neighbors[3*k:4*k] = rolled[j,i+1:i+1+k]
    neighbors[-1] = rolled[j,i]
    return neighbors

# @nb.njit(nb.int32[:](nb.int32, nb.int32[:,:], nb.int32, nb.int32, nb.int32))
def vonNeumann_outer(L, grid, j, i, k):
    rolled = np.roll(grid, shift=(int(L/2)-j, int(L/2)-i), axis=(0,1))
    j = int(L//2)
    i = int(L//2)
    neighbors = np.zeros(4*k+4*k*k)
    neighbors[   :k]   = rolled[j-k:j,i]
    neighbors[  k:2*k] = rolled[j+1:j+1+k,i]
    neighbors[2*k:3*k] = rolled[j,i-k:i]
    neighbors[3*k:4*k] = rolled[j,i+1:i+1+k]
    return neighbors

nbc = {1:Moore_outer, 2:Moore_inner, 3:vonNeumann_outer, 4:vonNeumann_inner}