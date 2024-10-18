#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:42:24 2024

@author: reinierramos
"""

import numpy as np
import itertools as itools
from numpy import random as nrand
from .ncadutils import applyDefect, updateGrid

rng = nrand.default_rng(17)

def solveNCAD(ncaActivity, defectIndices=None, dq=1/3, df=1/3, tRefrac=1):
    """
    Solves the spatiotemporal snapshots of a Discrete NCA of the same shape
    as ncaActivity. The CA is initialized with random distribution specified
    by dq and df, while ncaActivity is interpreted as the probability to fire.

    Parameters
    ----------
    ncaActivity : (duration, L, L)
        Activity of the Neuronal CA that is interpreted as the probability
        to fire.
    defectIndices : list of tuples, default is None
        List of indices of spike-defective neurons.
    dq : float, default is 1/3
        Initial density of "Q" cells in the CA.
        Must be between [0,1].
        `Note: total must be dq+df+dr=1.`
    df : float, default is 1/3
        Initial density of "F" cells in the CA.
        Must be between [0,1].
        `Note: total must be dq+df+dr=1`.
    tRefrac : int, default is 1
        Refractory period. 
        Number of timesteps that a neuron will stay "R".
        Must be nonzero.

    Returns
    -------
    dsoln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of Discrete NCA.

    See Also
    --------
    NeuronalCA.solveNCA -> soln = ncaActivity
    BriansBrain.solveBB

    """
    duration, L, _ = ncaActivity.shape
    dr = 1-(dq+df)
    
    dgrid = rng.choice([0,1,2], size=(L,L), p=(dq,df,dr)).astype(np.int32)
    grid_coords = list(itools.product(range(L), repeat=2))
    
    rgrid = np.zeros((L,L), dtype=int)
    rgrid[dgrid==2] = 1
    
    if defectIndices:
        dgrid = applyDefect(dgrid, defectIndices)
    
    dsoln = np.zeros(ncaActivity.shape, dtype=np.int32)
    dsoln[0] = dgrid
    
    pgrid = rng.random(size=ncaActivity.shape)
    
    propsCA = {'L': L, 'pgrid':pgrid, 'ncaActivity':ncaActivity,
              'timeRefrac':tRefrac,  'gridRefrac':rgrid}
    for t in range(1, duration):
        dgrid = updateGrid(L, t, dgrid, grid_coords, propsCA)
        if defectIndices: dgrid = applyDefect(dgrid, defectIndices)
        rgrid[dgrid==2] += 1
        rgrid[dgrid==0] = 0
        rgrid[dgrid==1] = 0
        propsCA.update({'gridRefrac':rgrid})
        dsoln[t,:,:] = dgrid
    return dsoln