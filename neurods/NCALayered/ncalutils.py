#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:00:17 2024

@author: reinierramos
"""

import numpy as np
from .ncalboundary import getNeighbors

def applyDefect(grid:list[float], defectIndex:list[int]) -> list[float]:
    """
    Applies spike-defect to the grid by setting the cells with indices
    specified by defectIndex to a state of 1 (maximum activation).

    """
    for layer,j,i in defectIndex:
        grid[layer,j,i] = 1
    return grid

def updateGrid(L:int, grid:list[float], grid_coords:list[int], propsCA:dict):
    """
    Updates the Layered Neuronal CA by applying activation equation.
    See Also: activationEquation(a_in, a0, a1, a2, nl)

    """
    a0, a1, a2 = propsCA.get('a0'), propsCA.get('a1'), propsCA.get('a2')
    nl = propsCA.get('nl')
    prev = grid.copy()
    for layer, j,i in grid_coords:
        a_in = np.mean(getNeighbors(propsCA, prev, layer, j, i))
        grid[layer,j,i] = activationFunction(a_in, a0, a1, a2, nl)
    return grid

def activationFunction(a_in:float, a0:float, a1:float, a2:float, 
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