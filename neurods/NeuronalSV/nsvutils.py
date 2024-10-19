#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:03:25 2024

@author: reinierramos
"""

import numpy as np
import networkx as nx

from scipy.spatial import SphericalVoronoi
import ai.cs as ac
from sphericalpolygon import Sphericalpolygon as csp
import itertools as it
from collections import defaultdict

def fibonacci_sphere(num_points:int, R:int=10) -> list[float]:
    """
    Generates points evenly distribute on a sphere's surface with radius R.
    See more information here: 
        https://extremelearning.com.au/evenly-distributing-points-on-a-sphere/
    
    """
    golden_angle = (3 - np.sqrt(5)) * np.pi
    theta = golden_angle * np.arange(num_points)
    points = np.zeros([num_points, 3])
    points[:,2] = np.linspace(1/num_points-1, 1-1/num_points, num_points)
    points[:,0] = np.sqrt(1-points[:,2] * points[:,2]) * np.cos(theta)
    points[:,1] = np.sqrt(1-points[:,2] * points[:,2]) * np.sin(theta)
    return R * points

def get_centroids(points:int, R:int=10) -> list[float]:
    """
    Finds the location of centroids of a Spherical Voronoi diagram.
    
    """
    sv = SphericalVoronoi(points, radius=R, center=[0,0,0])
    sv.sort_vertices_of_regions()
    centroids= np.zeros([len(points),3])
    for region in range(len(sv.regions)):
        vertices = [sv.vertices[r] for r in sv.regions[region]]
        vertices_sp = np.array([list(ac.cart2sp(v[0], v[1], v[2])) for v in vertices])
        polygon = csp.from_array(vertices_sp[:,1:3] * 180/np.pi)
        centroids[region] = (polygon.centroid(R)[0].to_value() * np.pi/180,
                             polygon.centroid(R)[1].to_value() * np.pi/180,
                             polygon.centroid(R)[2] * np.abs(polygon.centroid(R)[2]-R))

    centroids = np.array(ac.sp2cart(centroids[:,2],centroids[:,0],centroids[:,1])).T
    R_centroids = R * np.array([c/np.linalg.norm(c) for c in centroids])
    return R_centroids

def get_adjacency(points:int, R:int=10) -> list[int]:
    """
    Returns the list of edges generated from the centroidal Spherical Voronoi.
    
    """
    sv = SphericalVoronoi(points, radius=R, center =[0,0,0])
    sv.sort_vertices_of_regions()
    vertices_to_region = defaultdict(list) # Vertices-region adj
    for vertex, region in enumerate(sv.regions):
        for r in region:
            vertices_to_region[r].append(vertex)
            
    adjacencies = defaultdict(set)  # Region-region adj
    for vertex, regions in vertices_to_region.items():
        for r1, r2 in it.combinations(regions, 2):
            adjacencies[r1].add(r2)
            adjacencies[r2].add(r1)
    return adjacencies

def updateGrid(N:int, grid:list[float], G, propsCA:dict):
    """
    Updates the states of N neurons in graph G by applying activation equation
    with grid as the states at t-1.
    See Also: activationEquation(a_in, a0, a1, a2, nl)

    """
    a0, a1, a2 = propsCA.get('a0'), propsCA.get('a1'), propsCA.get('a2')
    nl = propsCA.get('nl')
    grid = np.array(list(nx.get_node_attributes(G, 'state').values()))
    for neuron in range(N):
        a_in = np.mean(getNeighbors(neuron, G))
        grid[neuron] = activationFunction(a_in, a0, a1, a2, nl)
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
    
def getNeighbors(neuron, G):
    """
    Returns the state of the neighbors of the neuron from its edges in G.
        
    """
    neighbor_list = np.array(list(G.neighbors(neuron))).astype(int)
    neighbor_states = np.array([G.nodes[n]['state'] for n in neighbor_list])
    return neighbor_states

def applyDefect(grid:list[float], defectIndex:list[int]) -> list[float]:
    """
    Applies spike-defect to the grid by setting the nodes with IDs 
    specified by defectIndex to a state of 1 (maximum activation).

    """
    for n in defectIndex:
        grid[n] = 1
    return grid
