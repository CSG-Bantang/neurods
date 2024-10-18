#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 01:01:31 2024

@author: reinierramos
"""

import numpy as np
from PIL import Image, ImageOps

import matplotlib as mpl
from matplotlib import pyplot as plt

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


def animateGOL(soln, out='animGOL.gif'):
    """
    Saves the spatiotemporal dynamics of GOL CA.

    Parameters
    ----------
    soln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of GOL CA.
    out : str, default is 'animGOL.gif'
        Output file name of the GIF.
        Must end with '.gif'
        
    """
    duration, L, _ = soln.shape
    resize = 200
    ims = [Image.fromarray(np.uint8(soln[i,:,:]*255)) for i in range(duration)]
    ims = [im.convert('P', palette=Image.ADAPTIVE, colors=2) for im in ims]
    ims = [ImageOps.contain(im, (resize,resize)) for im in ims]
    ims[0].save(fp=out, format='gif', append_images=ims, save_all=True, 
                duration=duration, loop=0
                )

def plotDensity(soln):
    """
    Plots the density of "alive" cells over time.

    Parameters
    ----------
    soln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of GOL CA.

    Returns
    -------
    fig : matplotlib.figure.Figure object
        Figure instance for which Phi(t) is plotted.
        Has all the attributes of matplotlib.figure.Figure
    ax : matplotlib.axes._axes.Axes object
        Axes instance for which Phi(t) is plotted.
        Has all the attributes of matplotlib.axes._axes.Axes
    
    fig and ax behaves similar to ``fig, ax = plt.subplots()``.

    """
    duration, L, _ = soln.shape
    N = L*L
    density = np.sum(soln, axis=(1,2)) / N
    fig, ax = plt.subplots(figsize=(6,5),
                           subplot_kw=dict(xlim=(-0.5, duration+0.5)
                                         , ylim=(-0.05, 1.05)
                                         , xlabel='Generation'
                                         , ylabel='Density of "Alive" Cells'))
    ax.plot(range(duration), density, color='darkgreen', lw=2, marker= 'o', 
            markersize=7, markerfacecolor='w')
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax