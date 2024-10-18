#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:07:25 2024

@author: reinierramos
"""

import numpy as np
import numba as nb

def getNeighbors(propsCA:dict, grid:list[float], layer:int, j:int, 
                 i:int) -> list[float]:
    """
    Returns neighbors of cell[j,i] from grid 
    given the following keys from propsCA:
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
        'L': int, lattice size
            L > 0.
        'layers' : int, default is 2
            Number of layers
    """
    L = propsCA.get('L')
    layers = propsCA.get('layers')
    system = propsCA.get('system')
    func = nbc.get(system)
    return func(L,grid,layer,j,i,layers)

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:,:], nb.int32, nb.int32, nb.int32, nb.int32))
def toroidal_Moore_inner(L, grid, layer, j, i, layers):
    neighbors = np.asarray([grid[layer, j-1, i-1],   grid[layer, j, i-1],   grid[layer, j-L+1, i-1],
                          grid[layer, j-1, i],     grid[layer, j, i],     grid[layer, j-L+1, i],
                          grid[layer, j-1, i-L+1], grid[layer, j, i-L+1], grid[layer, j-L+1, i-L+1]])
    if layers == 2:
        layerNeighbors = np.asarray([grid[layer-1, j, i]])
    else:
        layerNeighbors = np.asarray([grid[layer-layers+1, j, i], grid[layer-1, j, i]])
    allNeighbors = np.empty(len(neighbors)+len(layerNeighbors)).astype(np.float32)
    allNeighbors[:len(neighbors)] = neighbors
    allNeighbors[len(neighbors):] = layerNeighbors
    return allNeighbors

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:,:], nb.int32, nb.int32, nb.int32, nb.int32))
def toroidal_Moore_outer(L, grid, layer, j, i, layers):
    neighbors = np.asarray([grid[layer, j-1, i-1],   grid[layer, j, i-1],   grid[layer, j-L+1, i-1],
                          grid[layer, j-1, i],                            grid[layer, j-L+1, i],
                          grid[layer, j-1, i-L+1], grid[layer, j, i-L+1], grid[layer, j-L+1, i-L+1]])
    if layers == 2:
        layerNeighbors = np.asarray([grid[layer-1, j, i]])
    else:
        layerNeighbors = np.asarray([grid[layer-layers+1, j, i], grid[layer-1, j, i]])
    allNeighbors = np.empty(len(neighbors)+len(layerNeighbors)).astype(np.float32)
    allNeighbors[:len(neighbors)] = neighbors
    allNeighbors[len(neighbors):] = layerNeighbors
    return allNeighbors

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:,:], nb.int32, nb.int32, nb.int32, nb.int32))
def toroidal_vonNeumann_outer(L, grid, layer, j, i, layers):
    neighbors = np.asarray([grid[layer, j, i-1],   grid[layer, j-1, i], 
                     grid[layer, j-L+1, i], grid[layer, j, i-L+1]])
    if layers == 2:
        layerNeighbors = np.asarray([grid[layer-1, j, i]])
    else:
        layerNeighbors = np.asarray([grid[layer-layers+1, j, i], grid[layer-1, j, i]])
    allNeighbors = np.empty(len(neighbors)+len(layerNeighbors)).astype(np.float32)
    allNeighbors[:len(neighbors)] = neighbors
    allNeighbors[len(neighbors):] = layerNeighbors
    return allNeighbors

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:,:], nb.int32, nb.int32, nb.int32, nb.int32))
def toroidal_vonNeumann_inner(L, grid, layer, j, i, layers):
    neighbors = np.asarray([grid[layer, j, i-1],   grid[layer, j-1, i], grid[layer, j,i],
                     grid[layer, j-L+1, i], grid[layer, j, i-L+1]])
    if layers == 2:
        layerNeighbors = np.asarray([grid[layer-1, j, i]])
    else:
        layerNeighbors = np.asarray([grid[layer-layers+1, j, i], grid[layer-1, j, i]])
    allNeighbors = np.empty(len(neighbors)+len(layerNeighbors)).astype(np.float32)
    allNeighbors[:len(neighbors)] = neighbors
    allNeighbors[len(neighbors):] = layerNeighbors
    return allNeighbors

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:,:], nb.int32, nb.int32, nb.int32, nb.int32))
def spherical_Moore_outer(L, grid, layer, j, i, layers):
    if j == 0:
        neighbors = np.empty(L-1+3).astype(np.float32)
        neighbors[:i] = grid[layer, j,:i]
        neighbors[i:L-1] = grid[layer, j,i+1:L]
        neighbors[L-1:] = [grid[layer, j+1,i-1], grid[layer, j+1,i], grid[layer, j+1,i-L+1]]
    elif j == L-1:
        neighbors = np.empty(L-1+3).astype(np.float32)
        neighbors[:i] = grid[layer, j,:i]
        neighbors[i:L-1] = grid[layer, j,i+1:L]
        neighbors[L-1:] = [grid[layer, j-1,i-1], grid[layer, j-1,i], grid[layer, j-1,i-L+1]]
    else:
        neighbors = np.asarray([grid[layer, j-1, i-1  ], grid[layer, j, i-1  ], grid[layer, j-L+1, i-1  ],
                              grid[layer, j-1, i    ],                        grid[layer, j-L+1, i    ],
                              grid[layer, j-1, i-L+1], grid[layer, j, i-L+1], grid[layer, j-L+1, i-L+1]])
    if layers == 2:
        layerNeighbors = np.asarray([grid[layer-1, j, i]])
    else:
        layerNeighbors = np.asarray([grid[layer-layers+1, j, i], grid[layer-1, j, i]])
    allNeighbors = np.empty(len(neighbors)+len(layerNeighbors)).astype(np.float32)
    allNeighbors[:len(neighbors)] = neighbors
    allNeighbors[len(neighbors):] = layerNeighbors
    return allNeighbors

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:,:], nb.int32, nb.int32, nb.int32, nb.int32))
def spherical_Moore_inner(L, grid, layer, j, i, layers):
    if j == 0:
        neighbors = np.empty(L+3).astype(np.float32)
        neighbors[:L] = grid[layer, j,:]
        neighbors[L:] = [grid[layer, j+1,i-1], grid[layer, j+1,i], grid[layer, j+1,i-L+1]]
    elif j == L-1:
        neighbors = np.empty(L+3).astype(np.float32)
        neighbors[:L] = grid[layer, j,:]
        neighbors[L:] = [grid[layer, j-1,i-1], grid[layer, j-1,i], grid[layer, j-1,i-L+1]]
    else:
        neighbors = np.asarray([grid[layer, j-1, i-1  ], grid[layer, j, i-1  ], grid[layer, j-L+1, i-1  ],
                              grid[layer, j-1, i    ], grid[layer, j, i    ], grid[layer, j-L+1, i    ],
                              grid[layer, j-1, i-L+1], grid[layer, j, i-L+1], grid[layer, j-L+1, i-L+1]])
    if layers == 2:
        layerNeighbors = np.asarray([grid[layer-1, j, i]])
    else:
        layerNeighbors = np.asarray([grid[layer-layers+1, j, i], grid[layer-1, j, i]])
    allNeighbors = np.empty(len(neighbors)+len(layerNeighbors)).astype(np.float32)
    allNeighbors[:len(neighbors)] = neighbors
    allNeighbors[len(neighbors):] = layerNeighbors
    return allNeighbors

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:,:], nb.int32, nb.int32, nb.int32, nb.int32))
def spherical_vonNeumann_outer(L, grid, layer, j, i, layers):
    if j == 0:
        neighbors = np.empty(L-1+1).astype(np.float32)
        neighbors[:i] = grid[layer, j,:i]
        neighbors[i:L-1] = grid[layer, j,i+1:L]
        neighbors[L-1:] = grid[layer, j+1,i]
    elif j == L-1:
        neighbors = np.empty(L-1+1).astype(np.float32)
        neighbors[:i] = grid[layer, j,:i]
        neighbors[i:L-1] = grid[layer, j,i+1:L]
        neighbors[L-1:] = grid[layer, j-1,i]
    else:
        neighbors = np.asarray([grid[layer, j, i-1],   grid[layer, j-1, i], 
                         grid[layer, j-L+1, i], grid[layer, j, i-L+1]])
    if layers == 2:
        layerNeighbors = np.asarray([grid[layer-1, j, i]])
    else:
        layerNeighbors = np.asarray([grid[layer-layers+1, j, i], grid[layer-1, j, i]])
    allNeighbors = np.empty(len(neighbors)+len(layerNeighbors)).astype(np.float32)
    allNeighbors[:len(neighbors)] = neighbors
    allNeighbors[len(neighbors):] = layerNeighbors
    return allNeighbors

@nb.njit(nb.float32[:](nb.int32, nb.float32[:,:,:], nb.int32, nb.int32, nb.int32, nb.int32))
def spherical_vonNeumann_inner(L, grid, layer, j, i, layers):
    if j == 0:
        neighbors = np.empty(L+1).astype(np.float32)
        neighbors[:L] = grid[layer, j,:]
        neighbors[L:] = grid[layer, j+1,i]
    elif j == L-1:
        neighbors = np.empty(L+1).astype(np.float32)
        neighbors[:L] = grid[layer, j,:]
        neighbors[L:] = grid[layer, j-1,i]
    else:
        neighbors = np.asarray([grid[layer, j, i-1],   grid[layer, j-1, i], grid[layer, j,i],
                         grid[layer, j-L+1, i], grid[layer, j, i-L+1]])
    if layers == 2:
        layerNeighbors = np.asarray([grid[layer-1, j, i]])
    else:
        layerNeighbors = np.asarray([grid[layer-layers+1, j, i], grid[layer-1, j, i]])
    allNeighbors = np.empty(len(neighbors)+len(layerNeighbors)).astype(np.float32)
    allNeighbors[:len(neighbors)] = neighbors
    allNeighbors[len(neighbors):] = layerNeighbors
    return allNeighbors

nbc = {1:toroidal_Moore_outer,       2:toroidal_Moore_inner, 
       3:toroidal_vonNeumann_outer,  4:toroidal_vonNeumann_inner,
       5:spherical_Moore_outer,      6:spherical_Moore_inner, 
       7:spherical_vonNeumann_outer, 8:spherical_vonNeumann_inner}