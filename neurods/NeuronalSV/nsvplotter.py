#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:06:32 2024

@author: reinierramos
"""

import numpy as np
from scipy.spatial import SphericalVoronoi, geometric_slerp
from mayavi import mlab
from PIL import Image
from matplotlib import pyplot as plt

import matplotlib as mpl
axislabelsFontsize     = 17
titleFontsize          = 15
ticklabelsFontsize     = 15
dpiSize                = 300
figureParameters = {  'axes.labelsize' : axislabelsFontsize
                    , 'axes.titlesize' : titleFontsize
                    , 'xtick.labelsize': ticklabelsFontsize
                    , 'ytick.labelsize': ticklabelsFontsize
                    , 'savefig.dpi'    : dpiSize
                    , 'image.origin'   : 'lower'
                    }
mpl.rcParams.update(figureParameters)

NCAcmap = plt.get_cmap('magma')

def plotActivation(x, y, show_diagonal=True):
    """
    Plotter function for normalized return map of activation equation.
    A dashed diagonal, y = x, line is shown in the background for 
    future analysis.

    Parameters
    ----------
    x : 1D array
        Input states x[t].
    y : array of same length as x
        Output states x[t+1].
    show_diagonal : bool, default is True
        If True, plots a dashed diagonal, y = x, in the background.

    See Also
    --------
    ncaReturnMap -> x, y
    
    Returns
    -------
    fig : matplotlib.figure.Figure object
        Figure instance for which return map is plotted.
        Has all the attributes of matplotlib.figure.Figure
    ax : matplotlib.axes._axes.Axes object
        Axes instance for which return map is plotted.
        Has all the attributes of matplotlib.axes._axes.Axes
    
    fig and ax behaves similar to ``fig, ax = plt.subplots()``.

    """
    fig, ax = plt.subplots(figsize=(6,6),
              subplot_kw=dict(xlim=(-0.05,1.05), ylim=(-0.05,1.05)
              , xlabel='Previous Steady-State, $a_{in}$'
              , ylabel='Next Steady-State, $a_{out}$'))
    if show_diagonal:      ax.plot(x, x, color='gray', ls='--', lw=2)
    ax.plot(x, y, color='k', lw=4)
    ax.locator_params(axis='both', tight=True, nbins=5)
    fig.tight_layout()
    return fig, ax

def plotDensity(soln:list[float]):
    """
    Plots the density of states over time.

    Parameters
    ----------
    soln : (duration, N) array
        Snapshots of the spatiotemporal dynamics of Neuronal SV.
    Returns
    -------
    fig : matplotlib.figure.Figure object
        Figure instance for which return map is plotted.
        Has all the attributes of matplotlib.figure.Figure
    ax : matplotlib.axes._axes.Axes object
        Axes instance for which return map is plotted.
        Has all the attributes of matplotlib.axes._axes.Axes
    
    fig and ax behaves similar to ``fig, ax = plt.subplots()``.    
    
    See Also
    --------
    solveNSV -> soln

    """
    duration, N = soln.shape
    density = np.mean(soln, axis=1)
    fig, ax = plt.subplots(figsize=(6,5),
              subplot_kw=dict(xlim=(-0.5, duration+0.5), ylim=(-0.05, 1.05)
              , xlabel='Generation'
              , ylabel=r'SV Average $\langle a \rangle$'))
    ax.plot(range(duration), density, color='darkgoldenrod', lw=2, 
            marker= 'o', markersize=7, markerfacecolor='w')
    ax.locator_params(axis='both', tight=True, nbins=5)
    fig.tight_layout()
    return fig, ax

def animateNSV(soln, edges, xyz, out='animNSV.gif'):
    """
    Saves the spatiotemporal dynamics of Neuronal Spherical Voronoi as GIF.

    Parameters
    ----------
    soln : (duration, N) array
        where N = L*L, the total number of neurons.
        Snapshots of the spatiotemporal dynamics of Neuronal SV.
    edges : (E, 2) array
        where E is the number of edges generated using Spherical 
        Voronoi algorithm.
    xyz : (N, 3) array
        x, y, z coordinates of the centroids generated using Spherical 
        Voronoi algorithm.
    out : str, default is 'animNSV.gif'
        Output file name of the GIF.
        Must end with '.gif'
    
    See Also
    --------
    solveNSV -> soln, edges, xyz
    scipy.spatial.SphericalVoronoi
        
    """
    duration, N = soln.shape
    
    imsoln = generateScenes(duration, soln, edges, xyz)
    ims = [Image.fromarray(np.uint8(imsoln[i]*255)) 
           for i in range(duration)]
    ims[0].save(fp=out, format='gif', append_images=ims, save_all=True, 
                duration=duration, loop=0)

def generateScenes(duration, soln, edges, xyz, 
                   showNetwork=False, showTessellation=False):
    """
    Plots the soln, edges, and xyz using mayavi.mlab methods. 

    Parameters
    ----------
    duration : int
        Number of timesteps to solve Neuronal SV.
    soln : (duration, N) array
        where N = L*L, the total number of neurons.
        Snapshots of the spatiotemporal dynamics of Neuronal SV.
    edges : (E, 2) array
        where E is the number of edges generated using Spherical 
        Voronoi algorithm.
    xyz : (N, 3) array
        x, y, z coordinates of the centroids generated using Spherical 
        Voronoi algorithm.
    showNetwork : bool, default is False
        If True, the edges will be shown in the mlab figure.
    showTessellation : bool, default is False
        If True, the xyz of Voronoi regions will be shown in the mlab figure.

    Returns
    -------
    imsoln : (duration, figureSize, figureSize, 4) array
        RGBA Rendering of all snapshots generated from soln.

    """
    sceneSize = 400
    imsoln = np.zeros((duration,sceneSize,sceneSize,4))
    R = 10
    mlab.options.offscreen = True
    fig = mlab.figure(size=(sceneSize,sceneSize))
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    mlab.mesh(R*x, R*y, R*z,color=(255/255,234/255,201/255), opacity=0.8, 
              figure=fig)
    
    if showTessellation:
        sv = SphericalVoronoi(xyz, radius=R, center =[0,0,0])
        sv.sort_vertices_of_regions()
        t_vals = np.linspace(0, 1, 200)
        for region in sv.regions:
            n = len(region)
            for i in range(n):
                start = sv.vertices[region][i]
                end = sv.vertices[region][(i + 1) % n]
                start=start/np.linalg.norm(start)
                end=end/np.linalg.norm(end)
                result = geometric_slerp(start, end, t_vals)
                mlab.plot3d(R*result[..., 0], R*result[..., 1], R*result[..., 2], 
                            line_width=20, color=(0,0,0), figure=fig)
            
    for t in range(duration):
        pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2], soln[t,:], 
                            scale_mode='vector', 
                            colormap='magma', vmax=1, vmin=0, resolution=40, 
                            figure=fig)
        if showNetwork:
            pts.mlab_source.dataset.lines = edges
            tube = mlab.pipeline.tube(pts, tube_radius=0.1)
            mlab.pipeline.surface(tube, color=(0.0, 0.0, 0.0), opacity=0.6)
        mlab.view(figure=fig)
        imgmap = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
        imgmap[:,:,3] = 1
        imsoln[t,:,:,:] = imgmap
    mlab.close(all=True)
    return imsoln