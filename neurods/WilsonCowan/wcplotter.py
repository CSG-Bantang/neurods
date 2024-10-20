#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:50:32 2024

@author: reinierramos
"""

import numpy as np
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

def plotCurrent(tList, I_elist, I_ilist):
    """
    Plots the total input current Iext in uA/cm^2 to the WC system.

    Parameters
    ----------
    tList : (d,) array
        Time points for which HH is evaluated.
    I_elist : (d,) or (P, d) array
        Total input current for excitatory neurons.
    I_ilist : (d,) or (P, d) array
        Total input current for inhibitory neurons.

    Returns
    -------
    fig : matplotlib.figure.Figure object
        Figure instance for which V(t) is plotted.
        Has all the attributes of matplotlib.figure.Figure
    ax : matplotlib.axes._axes.Axes object
        Axes instance for which V(t) is plotted.
        Has all the attributes of matplotlib.axes._axes.Axes
    
    fig and ax behaves similar to ``fig, ax = plt.subplots()``.

    """
    _max = np.max((I_elist,I_ilist))
    _min = np.min((I_elist,I_ilist))
    ti, tf = tList[0], tList[-1]
    fig, ax = plt.subplots(figsize=(6,5),
              subplot_kw=dict(xlim=(ti-0.5, tf+0.5)
            , ylim=(_min-0.05,_max+0.05), xlabel='Time, in ms'
    , ylabel=r'Input Current $I_{\rm ext}~\left( \frac{\mu A}{cm^2}\right)$'))
    
    ax.plot(tList, I_elist, color='darkgreen', lw=2, alpha=0.3, label='Excitatory')
    ax.plot(tList, I_ilist, color='firebrick', lw=2, alpha=0.3, label='Inhibitory')
    ax.locator_params(axis='both', tight=True, nbins=5)
    plt.legend(fontsize=12, loc=1)
    return fig, ax

def plotDensity(soln, t):
    """
    Plots the density of excitatory and inhibitory neurons over time.

    Parameters
    ----------
    soln : (2, d) or (2, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (E, I) for all t in tList.

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
    duration = t[-1]
    densityE, densityI = soln
    fig, ax = plt.subplots(figsize=(6,5),
                           subplot_kw=dict(xlim=(-0.5, duration+0.5)
                                         , ylim=(-0.05, 1.05)
                                         , xlabel='Time in ms'
                                         , ylabel='Density of Neurons'))
    ax.plot(t, densityE, color='darkgreen', lw=2, label='Excitatory')
    ax.plot(t, densityI, color='firebrick', lw=2, label='Inhibitory')
    ax.locator_params(axis='both', tight=True, nbins=5)
    plt.legend(fontsize=12, loc=1)
    return fig, ax

def plotDensityPhaseSpace(soln):
    """
    Plots the limit cycle of excitatory against inhibitory neuron density.

    Parameters
    ----------
    soln : (2, d) or (2, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (E, I) for all t in tList.

    Returns
    -------
    fig : matplotlib.figure.Figure object
        Figure instance for which V(t) is plotted.
        Has all the attributes of matplotlib.figure.Figure
    ax : matplotlib.axes._axes.Axes object
        Axes instance for which V(t) is plotted.
        Has all the attributes of matplotlib.axes._axes.Axes
    
    fig and ax behaves similar to ``fig, ax = plt.subplots()``.

    """
    E, I = soln
    fig, ax = plt.subplots(figsize=(6,6),
                           subplot_kw=dict(xlim=(-0.05,1.05)
                                         , ylim=(-0.05,1.05)
                                         , xlabel='Inhibitory I(t)'
                                         , ylabel='Excitatory E(t)'))
    ax.plot(I, E, color='dodgerblue', lw=2)
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax