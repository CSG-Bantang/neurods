#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:16:25 2024

@author: reinierramos
"""

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from matplotlib import cm, animation, colors

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
    return fig, ax

def plotDensity(soln:list[float]):
    """
    Plots the density of states over time.

    Parameters
    ----------
    soln : (duration, layers, L, L) array
        Snapshots of the spatiotemporal dynamics of Neuronal CA.
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
    solveNCAL -> soln

    """
    duration, layers, L, _ = soln.shape
    N = L*L
    density = np.mean(soln, axis=(1,2,3))
    densLay = np.mean(soln, axis=(2,3))
    fig, ax = plt.subplots(figsize=(6,5),
              subplot_kw=dict(xlim=(-0.5, duration+0.5), ylim=(-0.05, 1.05)
              , xlabel='Generation'
              , ylabel=r'CA Average $\langle a \rangle$'))
    for layer in range(layers):
        ax.plot(range(duration), densLay[:,layer], lw=1, 
                label=f'layer {layer+1}', alpha=0.8)
    ax.plot(range(duration), density, color='black', lw=4, marker= 'o', 
            markersize=7, markerfacecolor='w', label='all layers')
    ax.locator_params(axis='both', tight=True, nbins=5)
    plt.legend(fontsize=12, loc=1)
    return fig, ax

def animateNCAL(soln, out='animNCAL.gif'):
    """
    Saves the spatiotemporal dynamics of Neuronal CA as GIF.

    Parameters
    ----------
    soln : (duration, layers, L, L) array
        Snapshots of the spatiotemporal dynamics of Layered NCA.
    out : str, default is 'animNCAL.gif'
        Output file name of the GIF.
        Must end with '.gif'
    
    See Also
    --------
    solveNCAL -> soln
        
    """
    duration, layers, L, _ = soln.shape
    resize = 200
    soln = np.hstack(tuple([soln[:,layer,:,:] for layer in range(layers)]))
    ims = [Image.fromarray(np.uint8(NCAcmap(soln[i,:,:])*255)) 
           for i in range(duration)]
    ims = [im.convert('P', palette=Image.ADAPTIVE, colors=100) for im in ims]
    ims = [ImageOps.contain(im, (resize,resize)) for im in ims]
    ims[0].save(fp=out, format='gif', append_images=ims, save_all=True, 
                duration=duration, loop=0)

def animate3DNCAL(soln, out='anim3DNCAL.gif'):
    """
    Saves the spatiotemporal dynamics of 3D Layered Neuronal CA as GIF.

    Parameters
    ----------
    soln : (duration, layers, L, L) array
        Snapshots of the spatiotemporal dynamics of Layered NCA.
    out : str, default is 'anim3DNCAL.gif'
        Output file name of the GIF.
        Must end with '.gif'
    
    See Also
    --------
    solveNCAL -> soln
    
    """
    duration, layers, L, _ = soln.shape
    cmap=cm.magma
    x,y,z = np.indices((layers+1,L+1,L+1))
    filled = np.ones((layers,L,L))
    norm=colors.Normalize(vmin = 0, vmax = 1, clip = False)
    facecolors = cmap(norm(soln[0,:,:,:]))
    fig = plt.figure(figsize=(7,7), layout='tight')
    ax = fig.add_subplot(projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    ax.voxels(x,y,z, filled, facecolors=facecolors, edgecolor='k',
              linewidth=0.01)
    
    ax.axis('off')
    ax.set_aspect('equal')
    anim = animation.FuncAnimation(fig, update, frames=range(duration),
                                   fargs=(soln, ax, cmap, norm, x,y,z,filled),
                                   interval=100, blit=True)
    anim.save(out, dpi=200)
    
def update(t:int, soln:list[float], ax, cmap, norm, x,y,z,filled):
    """
    Update function used for generating artists for FuncAnimation.

    """
    facecolors = cmap(norm(soln[t,:,:,:]))
    im = ax.voxels(x,y,z, filled, facecolors=facecolors, edgecolor='k',
              linewidth=0.01)
    return list(im.values())