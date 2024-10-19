#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:36:21 2024

@author: reinierramos
"""

import numpy as np
from matplotlib import pyplot as plt

from PIL import Image, ImageOps
from matplotlib import colors as mplc

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

Q,F,R = range(3)
QFRcmap = mplc.ListedColormap(['black','khaki', 'rebeccapurple'], N=3)

def plotDensity(soln:list[float]):
    """
    Plots the density of Q, F, and R states over time.

    Parameters
    ----------
    soln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of Brian's Brain CA.
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
    solveBB -> soln

    """
    duration, L, _ = soln.shape
    N = L*L
    densityF = np.sum(soln==F, axis=(1,2)) / N
    densityR = np.sum(soln==R, axis=(1,2)) / N
    densityQ = np.sum(soln==Q, axis=(1,2)) / N
    fig, ax = plt.subplots(figsize=(6,5),
                           subplot_kw=dict(xlim=(-0.5, duration+0.5)
                                         , ylim=(-0.05, 1.05)
                                         , xlabel='Generation'
                                         , ylabel='Density of Cells'))
    ax.plot(range(duration), densityR, color='rebeccapurple', lw=1, 
            label='R', ls='--')
    ax.plot(range(duration), densityQ, color='black', ls='--', marker= 'd',
            markersize=7, markerfacecolor='None', label='Q', alpha=0.6)
    ax.plot(range(duration), densityF, color='darkgoldenrod', marker= 'o', 
            markersize=10, markerfacecolor='w', label='F', lw=2)
    ax.locator_params(axis='both', tight=True, nbins=5)
    plt.legend(fontsize=12, loc=1)
    return fig, ax

def animateBB(soln, out='animBB.gif'):
    """
    Saves the spatiotemporal dynamics of Brian's Brain as GIF.

    Parameters
    ----------
    soln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of BB CA.
    out : str, default is 'animBB.gif'
        Output file name of the GIF.
        Must end with '.gif'
    
    See Also
    --------
    solveBB -> soln
    
    """
    duration, L, _ = soln.shape
    resize = 200
    ims = [Image.fromarray(np.uint8(QFRcmap(soln[i,:,:]/2)*255)) 
           for i in range(duration)]
    ims = [im.convert('P', palette=Image.ADAPTIVE, colors=3) for im in ims]
    ims = [ImageOps.contain(im, (resize,resize)) for im in ims]
    ims[0].save(fp=out, format='gif', append_images=ims, save_all=True, 
                duration=duration, loop=0)