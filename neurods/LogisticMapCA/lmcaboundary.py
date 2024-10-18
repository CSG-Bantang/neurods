#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:24:45 2024

@author: reinierramos
"""

import numpy as np
import numba as nb

def getNeighbors(propsCA:dict, grid:list[float], j:int, 
                 i:int) -> list[float]:
    """
    Returns neighbors of the cell grid[j,i] given the following 
    keys from propsCA:
        'L': int, lattice size
            L > 0.
        'system' : int, default is 1
            Neighborhood boundary condition.
            1: toroidal,  outer-totalistic Moore
            2: toroidal,  inner-totalistic Moore
            3: toroidal,  outer-totalistic von Neumann
            4: toroidal,  outer-totalistic von Neumann
            5: spherical, outer-totalistic Moore
            6: spherical, inner-totalistic Moore
            7: spherical, outer-totalistic von Neumann
            8: spherical, inner-totalistic von Neumann
        
    """
    L = propsCA.get('L')
    system = propsCA.get('system')
    func = nbc.get(system)
    return func(L,grid,j,i)

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:], nb.int32, nb.int32))
def toroidal_Moore_inner(L, grid, j, i):
    return np.array([grid[j-1, i-1],   grid[j, i-1],   grid[j-L+1, i-1],
                     grid[j-1, i],     grid[j, i],     grid[j-L+1, i],
                     grid[j-1, i-L+1], grid[j, i-L+1], grid[j-L+1, i-L+1]])

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:], nb.int32, nb.int32))
def toroidal_Moore_outer(L, grid, j, i):
    return np.array([grid[j-1, i-1],   grid[j, i-1],   grid[j-L+1, i-1],
                     grid[j-1, i],                     grid[j-L+1, i],
                     grid[j-1, i-L+1], grid[j, i-L+1], grid[j-L+1, i-L+1]])

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:], nb.int32, nb.int32))
def toroidal_vonNeumann_outer(L, grid, j, i):
    return np.array([grid[j, i-1],   grid[j-1, i], 
                     grid[j-L+1, i], grid[j, i-L+1]])

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:], nb.int32, nb.int32))
def toroidal_vonNeumann_inner(L, grid, j, i):
    return np.array([grid[j, i-1],   grid[j-1, i], grid[j,i],
                     grid[j-L+1, i], grid[j, i-L+1]])

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:], nb.int32, nb.int32))
def spherical_Moore_outer(L, grid, j, i):
    if j == 0:
        neighbors = np.empty(L-1+3).astype(np.float32)
        neighbors[:i] = grid[j,:i]
        neighbors[i:L-1] = grid[j,i+1:L]
        neighbors[L-1:] = [grid[j+1,i-1], grid[j+1,i], grid[j+1,i-L+1]]
    elif j == L-1:
        neighbors = np.empty(L-1+3).astype(np.float32)
        neighbors[:i] = grid[j,:i]
        neighbors[i:L-1] = grid[j,i+1:L]
        neighbors[L-1:] = [grid[j-1,i-1], grid[j-1,i], grid[j-1,i-L+1]]
    else:
        neighbors = np.array([grid[j-1, i-1  ], grid[j, i-1  ], grid[j-L+1, i-1  ],
                              grid[j-1, i    ],                 grid[j-L+1, i    ],
                              grid[j-1, i-L+1], grid[j, i-L+1], grid[j-L+1, i-L+1]])
    return neighbors

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:], nb.int32, nb.int32))
def spherical_Moore_inner(L, grid, j, i):
    if j == 0:
        neighbors = np.empty(L+3).astype(np.float32)
        neighbors[:L] = grid[j,:]
        neighbors[L:] = [grid[j+1,i-1], grid[j+1,i], grid[j+1,i-L+1]]
    elif j == L-1:
        neighbors = np.empty(L+3).astype(np.float32)
        neighbors[:L] = grid[j,:]
        neighbors[L:] = [grid[j-1,i-1], grid[j-1,i], grid[j-1,i-L+1]]
    else:
        neighbors = np.array([grid[j-1, i-1  ], grid[j, i-1  ], grid[j-L+1, i-1  ],
                              grid[j-1, i    ], grid[j, i    ], grid[j-L+1, i    ],
                              grid[j-1, i-L+1], grid[j, i-L+1], grid[j-L+1, i-L+1]])
    return neighbors

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:], nb.int32, nb.int32))
def spherical_vonNeumann_outer(L, grid, j, i):
    if j == 0:
        neighbors = np.empty(L-1+1).astype(np.float32)
        neighbors[:i] = grid[j,:i]
        neighbors[i:L-1] = grid[j,i+1:L]
        neighbors[L-1:] = grid[j+1,i]
    elif j == L-1:
        neighbors = np.empty(L-1+1).astype(np.float32)
        neighbors[:i] = grid[j,:i]
        neighbors[i:L-1] = grid[j,i+1:L]
        neighbors[L-1:] = grid[j-1,i]
    else:
        neighbors = np.array([grid[j, i-1],   grid[j-1, i], 
                         grid[j-L+1, i], grid[j, i-L+1]])
    return neighbors

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:], nb.int32, nb.int32))
def spherical_vonNeumann_inner(L, grid, j, i):
    if j == 0:
        neighbors = np.empty(L+1).astype(np.float32)
        neighbors[:L] = grid[j,:]
        neighbors[L:] = grid[j+1,i]
    elif j == L-1:
        neighbors = np.empty(L+1).astype(np.float32)
        neighbors[:L] = grid[j,:]
        neighbors[L:] = grid[j-1,i]
    else:
        neighbors = np.array([grid[j, i-1],   grid[j-1, i], grid[j,i],
                         grid[j-L+1, i], grid[j, i-L+1]])
    return neighbors

nbc = {1:toroidal_Moore_outer,       2:toroidal_Moore_inner, 
       3:toroidal_vonNeumann_outer,  4:toroidal_vonNeumann_inner,
       5:spherical_Moore_outer,      6:spherical_Moore_inner, 
       7:spherical_vonNeumann_outer, 8:spherical_vonNeumann_inner}