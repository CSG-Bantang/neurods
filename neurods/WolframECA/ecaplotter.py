#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:16:53 2024

@author: reinierramos
"""

import numpy as np
from matplotlib import pyplot as plt
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

def plotECA(soln):
    cmap = mplc.ListedColormap(['black', 'steelblue'])
    fig, ax = plt.subplots()
    ax.imshow(soln, vmin=0, vmax=1, cmap=cmap, origin='upper')
    ax.axis('off')
    fig.tight_layout()
    return fig, ax

def plotDensity(soln):
    duration, L = soln.shape
    density = np.sum(soln, axis=1) / L
    fig, ax = plt.subplots(figsize=(6,5),
                           subplot_kw=dict(xlim=(-0.5, duration+0.5)
                                         , ylim=(-0.05, 1.05)
                                         , xlabel='Generation'
                                         , ylabel='Density of Cells'))
    ax.plot(range(duration), density, color='darkgoldenrod', marker= 'o', 
            markersize=4, label='F', lw=1)
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax