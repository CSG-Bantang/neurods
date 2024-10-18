#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:20:34 2024

@author: reinierramos
"""

import numpy as np
from .lmcaboundary import getNeighbors

def updateGrid(L:int, grid:list[float], grid_coords:list[int], 
               propsCA:dict) -> list[float]:
    """
    Updates the LMCA grid applying logisticEquation.

    """
    r = propsCA.get('r')
    prev = grid.copy()
    for j,i in grid_coords:
        xin = np.mean(getNeighbors(propsCA, prev, j, i))
        grid[j,i] = logisticEquation(r, xin)
    return grid

def logisticEquation(r:float=1.0, xt:float|list=0.5) -> float|list:
    """
    Calculates the next value x(t+1) according to a logistic growth rate r
    given the current value xt.

    """
    return r*xt*(1-xt)