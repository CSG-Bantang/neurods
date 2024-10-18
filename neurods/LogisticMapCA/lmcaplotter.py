#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:32:57 2024

@author: reinierramos
"""

import numpy as np
from matplotlib import pyplot as plt

from PIL import Image, ImageOps
from matplotlib import cm, animation, colors
from mpl_toolkits.mplot3d import axes3d

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

LCAcmap = plt.get_cmap('magma')

def plotXvsT(x, t):
    """
    Plotter function for normalized steady-state x(t).

    Parameters
    ----------
    x : 1D array
        Values of x(t).
    t : array of same length as x
        Time points for which LM is evaluated.

    See Also
    --------
    solveLM -> xList, tList
    
    Returns
    -------
    fig : matplotlib.figure.Figure object
        Figure instance for which x(t) is plotted.
        Has all the attributes of matplotlib.figure.Figure
    ax : matplotlib.axes._axes.Axes object
        Axes instance for which x(t) is plotted.
        Has all the attributes of matplotlib.axes._axes.Axes
    
    fig and ax behaves similar to ``fig, ax = plt.subplots()``.

    """
    ti, tf = t[0], t[-1]
    fig, ax = plt.subplots(figsize=(6,5),
                            subplot_kw=dict(xlim=(ti-0.5, tf+0.5)
                                          , ylim=(-0.05,1.05)
                                          , xlabel='Timestep, t'
                                          , ylabel='Steady-state, x(t)'))
    ax.plot(t, x, color='k', lw=1, marker='d', markersize=8, markerfacecolor='white')
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax

def plotReturnMap(x, y, show_diagonal=True):
    """
    Plotter function for normalized return map of logistic equation.
    A dashed diagonal, y = x, line is shown in the background for future analysis.

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
    logisticReturnMap -> x, y

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
                           subplot_kw=dict(xlim=(-0.05,1.05)
                                         , ylim=(-0.05,1.05)
                                         , xlabel='Previous Steady-State, $x_{t}$'
                                         , ylabel='Next Steady-State, $x_{t+1}$'))
    if show_diagonal:      ax.plot(x, x, color='gray', ls='--', lw=2)
    ax.plot(x, y, color='k', lw=4)
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax

def plotDensity(soln:list[float]):
    """
    Plots the density of states over time.

    Parameters
    ----------
    soln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of Logistic CA.
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
    solveLMCA -> soln

    """
    duration, L, _ = soln.shape
    N = L*L
    density = np.mean(soln, axis=(1,2))
    fig, ax = plt.subplots(figsize=(6,5),
                           subplot_kw=dict(xlim=(-0.5, duration+0.5)
                                         , ylim=(-0.05, 1.05)
                                         , xlabel='Generation'
                                         , ylabel=r'CA Average $\langle x \rangle$'))
    ax.plot(range(duration), density, color='darkgoldenrod', lw=2, marker= 'o', markersize=7, markerfacecolor='w')
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax

def animateLMCA(soln, out='animLMCA.gif'):
    """
    Saves the spatiotemporal dynamics of Logistic CA as GIF.

    Parameters
    ----------
    soln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of LMCA.
    out : str, default is 'animLMCA.gif'
        Output file name of the GIF.
        Must end with '.gif'
    
    See Also
    --------
    solveLMCA -> soln
        
    """
    duration, L, _ = soln.shape
    resize = 200
    ims = [Image.fromarray(np.uint8(LCAcmap(soln[i,:,:])*255)) for i in range(duration)]
    ims = [im.convert('P', palette=Image.ADAPTIVE, colors=100) for im in ims]
    ims = [ImageOps.contain(im, (resize,resize)) for im in ims]
    ims[0].save(fp=out, format='gif', append_images=ims, save_all=True, 
                duration=duration, loop=0)

def animate3DLMCA(soln, out='anim3DLMCA.gif'):
    """
    Saves the spatiotemporal dynamics of Spherical Logistic CA as GIF.
    Use when specifying system = 5, 6, 7, or 8.

    Parameters
    ----------
    soln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of LMCA.
    out : str, default is 'anim3DLMCA.gif'
        Output file name of the GIF.
        Must end with '.gif'
    
    See Also
    --------
    solveLMCA -> soln
    
    """
    duration, L, _ = soln.shape
    fig = plt.figure(figsize=(7,7), layout='tight')
    ax = fig.add_subplot(projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    u, v = np.mgrid[0:np.pi:complex(0,L), 0:2*np.pi:complex(0,L)]
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    cmap=cm.magma
    norm=colors.Normalize(vmin = 0,
                          vmax = 1, clip = False)
    im = ax.plot_surface(x,y,z, cmap=cmap, rstride=1, cstride=1,
                         linewidth=0.01, facecolors=cmap(norm(soln[0,:,:])))
    ax.axis('off')
    ax.set_xlim(np.array([-1,1])*.6)
    ax.set_ylim(np.array([-1,1])*.65)
    ax.set_zlim(np.array([-1,1])*.6)
    ax.set_aspect('equal')
    anim = animation.FuncAnimation(fig, update, frames=range(duration),
                                   fargs=(soln, im, cmap, norm, L),
                                   interval=100, blit=True)
    anim.save(out, dpi=200)

def update(t:int, soln:list[float], im, cmap, norm, L:int):
    """
    Update function used for generating artists for FuncAnimation.

    """
    im.set_facecolors(cmap(norm(soln[t,:,:]))[:-1, :-1].reshape(L-1*L-1, 4))
    return im,