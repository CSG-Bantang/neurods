#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 23:05:09 2024

@author: reinierramos
"""

from matplotlib import pyplot as plt
from matplotlib import colors as mplc
import numpy as np

import matplotlib as mpl
axislabelsFontsize     = 17
titleFontsize          = 15
ticklabelsFontsize     = 15
dpiSize                = 300
colorbarlabelsFontsize = 15
figureParameters = {  'axes.labelsize' : axislabelsFontsize
                    , 'axes.titlesize' : titleFontsize
                    , 'xtick.labelsize': ticklabelsFontsize
                    , 'ytick.labelsize': ticklabelsFontsize
                    , 'savefig.dpi'    : dpiSize
                    , 'image.origin'   : 'lower'
                    }
mpl.rcParams.update(figureParameters)

def colorbarActivity():
    plt.ioff()
    fig, ax = plt.subplots(subplot_kw=dict(aspect='equal', 
                                           xticks=[], xticklabels=[], 
                                           yticks=[], yticklabels=[]))
    im      = ax.imshow(np.zeros((10,10), dtype=np.float32), 
                        cmap='magma', vmin=0, vmax=1)
    cbar    = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_ticks(ticks=[0, 0.5, 1], labels=['0.0', '0.5', '1.0'], 
                   fontsize=ticklabelsFontsize)
    cbar.set_label(r'Neuronal Activity, $a$', rotation = 270, 
                   labelpad = 15, fontsize = colorbarlabelsFontsize)
    ax.remove()
    return fig

def colorbarSteadyState():
    plt.ioff()
    fig, ax = plt.subplots(subplot_kw=dict(aspect='equal', 
                                           xticks=[], xticklabels=[], 
                                           yticks=[], yticklabels=[]))
    im      = ax.imshow(np.zeros((10,10), dtype=np.float32), 
                        cmap='magma', vmin=0, vmax=1)
    cbar    = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_ticks(ticks=[0, 0.5, 1], labels=['0.0', '0.5', '1.0'], 
                   fontsize=ticklabelsFontsize)
    cbar.set_label(r'Average steady-state, $\langle a \rangle_{\infty}$', 
                   rotation = 270, labelpad = 15, 
                   fontsize = colorbarlabelsFontsize)
    ax.remove()
    return fig

def colorbarQFR():
    RQFcmap = mplc.ListedColormap(['rebeccapurple', 'black','khaki'])
    R,Q,F   = range(3)
    plt.ioff()
    fig, ax = plt.subplots(subplot_kw=dict(aspect='equal', 
                                           xticks=[], xticklabels=[], 
                                           yticks=[], yticklabels=[]))
    im = ax.imshow(np.zeros((10,10), dtype=np.float32), 
                   cmap=RQFcmap, vmin=R, vmax=F)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_ticks(ticks=[0, 1, 2], labels=['R', 'Q', 'F'], 
                   fontsize=ticklabelsFontsize)
    cbar.set_label('Neuronal Discrete State', rotation = 270, 
                   labelpad = 15, fontsize = colorbarlabelsFontsize)
    ax.remove()
    return fig