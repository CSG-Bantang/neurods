#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:56:56 2024

@author: reinierramos
"""

STATES = 3
Q,F,R = range(STATES)

def applyDefect(grid:list[float], defectIndex:list[int]) -> list[float]:
    """
    Applies spike-defect to the grid by setting the cells with indices
    specified by defectIndex to a state of 1 (maximum activation).

    """
    for layer,j,i in defectIndex:
        grid[layer,j,i] = 1
    return grid

def updateGrid(L:int, t:int, grid:list[int], grid_coords:list[int], 
               propsCA:dict) -> list[float]:
    """
    Updates the Discrete NCA by applying discreteRules.

    """
    pgrid = propsCA.get('pgrid')
    ncaActivity = propsCA.get('ncaActivity')
    timeRefrac = propsCA.get('timeRefrac')
    gridRefrac = propsCA.get('gridRefrac')
    layers = propsCA.get('layers')
    for layer, j,i in grid_coords:
        cell = grid[layer,j,i]
        firingCondition = (pgrid[t,layer,j,i] <= ncaActivity[t,layer,j,i])
        refracCondition = (gridRefrac[layer,j,i]<timeRefrac)
        grid[layer,j,i] = discreteRules(cell, firingCondition, refracCondition)
    return grid

def discreteRules(cell:int, firingCondition:bool, refracCondition:bool):
    """
    Discrete Transition Rules:
        if cell=F,                           cell->R
        if cell=R and refracCondition=False, cell->Q
        if cell=Q and firingCondition=True,  cell->F
        if cell=R and refracCondition=True,  cell stays R 
    Refractory condition is given by tRefrac.
    Firing condition is given by ncaActivity.
    
    See Also
    --------
    solveNCAD(ncaActivity, tRefrac=1)

    """
    return R*(cell==F) + Q*(not refracCondition)*(cell==R) + F*firingCondition*(cell==Q) + \
            R*refracCondition*(cell==R)