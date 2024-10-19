#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:04:31 2024

@author: reinierramos
"""

import numpy as np
import networkx as nx
from .nsvutils import (fibonacci_sphere, get_centroids, 
                       get_adjacency, updateGrid, applyDefect)
from numpy import random as nrand
rng = nrand.default_rng(17)

def solveNSV(duration=30, L=4, a0=0.0, a1=0.8, a2=0.9, nl=1.0, 
             init='uniform', defect=0, **kwargs):
    """
    Solves the spatiotemporal snapshots of a Neuronal Spherical Voronoi
    with neuronal population of N = L*L. The SV is initialized with 
    random distribution specified by init.

    Parameters
    ----------
    duration : int, default is 30
        Number of timesteps to solve Neuronal SV.
        Must be nonzero.
    L : int, default is 4
        Lattice size for the NSV.
        Must be greater than 0.
    a0 : float, default is 0.0
        Minimum input threshold
        Must be between [0,1].
    a1 : float, default is 0.8
        Maximum input threshold
        Must be between [0,1].
    a2 : float, default is 0.9
        Maximum output threshold
        Must be between [0,1].
    nl : float, default is 1.0
        Nonlinearity of activation profile.
        Must be greater than 0.
    init : str, default is 'uniform'
        Random distribution to initialize Neuronal SV.
        Accepted values: 'uniform' and 'beta'.
        If 'beta', additional `kwargs must be provided, 
        either ``(a=, b=)`` or ``(mu=, nu=)``.
    defect : float, default is 0
        Density of spike-defective neurons.
        Must be between [0,1].
    **kwargs : dict
        Additional arguments needed to specify shape parameters of beta 
        distribution. If mu and nu are specified, then the parameters a
        and b are automatically computed. Either choose ``(a=, b=)`` or 
        ``(mu=, nu=)``.
        a: float, must be nonzero, a=mu*nu
        b: float, must be nonzero, b=(1-mu)*nu
        mu: float, mean, Must be in (0,1).
        nu: float, precision of the mean, "sample size" in Bayes theorem.

    Returns
    -------
    soln : (duration, N) array
        where N = L*L, the total number of neurons.
        Snapshots of the spatiotemporal dynamics of Neuronal SV.
    edges : (E, 2) array
        where E is the number of edges generated using Spherical 
        Voronoi algorithm.
    xyz : (N, 3) array
        x, y, z coordinates of the centroids generated using Spherical 
        Voronoi algorithm.
    defectIndices : optional, list with length int(defect*L*L)
        Returns list of tuples indicating the node IDs of spike-defective 
        neurons if defect is nonzero.

    """
    N = L*L
    fiboverts = fibonacci_sphere(N)
    centroverts = get_centroids(fiboverts)
    edgelist = get_adjacency(centroverts)
    G = nx.Graph(edgelist)
    
    nodes = np.array(sorted(G.nodes()))
    edges = np.array(list(G.edges()))
    xyz = np.asarray(centroverts)
    
    if init=='beta' and kwargs.get('a') and kwargs.get('b'):
        a, b = kwargs.get('a'), kwargs.get('b')
        grid = rng.beta(a, b, size=N).astype(np.float32)
    if init=='beta' and kwargs.get('mu') and kwargs.get('nu'):
        mu, nu = kwargs.get('mu'), kwargs.get('nu')
        a, b = mu*nu, (1-mu)*nu
        grid = rng.beta(a, b, size=N).astype(np.float32)
    if init=='uniform':
        grid = rng.random(size=N, dtype=np.float32)
    
    if defect:
        rng.shuffle(defectIndex := np.asarray(range(N)))
        grid = applyDefect(grid, defectIndex[:int(defect*N)])
    
    state_init = dict(zip(nodes, grid))
    nx.set_node_attributes(G, state_init, 'state')
    grid = np.array(list(nx.get_node_attributes(G, 'state').values()))
    
    soln = np.zeros((duration+1, N))
    soln[0,:] = grid
    
    propsCA = {'a0':a0, 'a1':a1, 'a2':a2, 'nl':nl}
    for t in range(duration):
        grid = updateGrid(N, grid, G, propsCA)
        if defect: grid = applyDefect(grid, defectIndex[:int(defect*N)])
        states = dict(zip(nodes, grid))
        nx.set_node_attributes(G, states, 'state')
        soln[t+1,:] = grid
    if defect:
        return soln, edges, xyz, defectIndex[:int(defect*N)]
    return soln, edges, xyz

def ncaReturnMap(a0:float, a1:float, a2:float, nl:float) -> tuple[list[float]]:
    """
    Input-output map for the activation equation 
    a[t+1] = a2(1 - (1 - (a[t]-a0)/(a1-a0))^b) for a1 > a[t] > a0,
    zero otherwise.

    Parameters
    ----------
    a0 : float
        Minimum input threshold
        Must be between [0,1].
    a1 : float
        Maximum input threshold
        Must be between [0,1].
    a2 : float
        Maximum output threshold
        Must be between [0,1].
    nl : float
        Nonlinearity of activation profile.
        Must be greater than 0.

    Returns
    -------
    x : (numpoints,) array
        Input states a_in = a[t].
    y : (numpoints,) array
        Output states a_out = a[t+1].
    
    """
    numpoints = 300
    x = np.linspace(0, 1, numpoints)
    y = np.array([activationEquation(a, a0, a1, a2, nl) for a in x])
    return x, y

def activationEquation(a_in:float, a0:float, a1:float, a2:float, 
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