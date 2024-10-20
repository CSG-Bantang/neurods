#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 21:35:36 2024

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

def plotVoltage(soln, tList):
    """
    Plots the membrane voltage Phi in mV versus time in ms.

    Parameters
    ----------
    soln : (2, d) or (2, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (Phi, n) for all t in tList.
    tList : (d,) array
        Time points for which ML is evaluated.

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
    ti, tf = tList[0], tList[-1]
    fig, ax = plt.subplots(figsize=(6,5),
                           subplot_kw=dict(xlim=(ti-0.5, tf+0.5)
                                         , ylim=(-70,40)
                                         , xlabel='Time, in ms'
                                         , ylabel='Voltage, in mV'))
    V, _ = soln
    if len(V.shape) == 2:
        for _i in range(V.shape[1]):
            if _i == 0:  ax.plot(tList, V[:,_i], color='k', lw=2)
            else:        ax.plot(tList, V[:,_i])
    elif len(V.shape) == 1:
        ax.plot(tList, V, color='k', lw=2)
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax

def plotChannels(soln, tList):
    """
    Plots the gating channel variables n over time t in ms.

    Parameters
    ----------
    soln : (2, d) or (2, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (Phi, n) for all t in tList.
    tList : (d,) array
        Time points for which ML is evaluated.

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
    ti, tf = tList[0], tList[-1]
    fig, ax = plt.subplots(figsize=(6,4),
                           subplot_kw=dict(xlim=(ti-0.5, tf+0.5)
                                         , ylim=(-0.05,1.05)
                                         , xlabel='Time, in ms'
                                         , ylabel='Gating variable'))
    _, n = soln    
    ax.plot(tList, n, color='goldenrod', lw=2, label='n(t)')
    plt.legend(fontsize=12, loc=1)
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax

def plotCurrent(tList, Ilist):
    """
    Plots the total input current Iext in uA/cm^2 to the ML system.

    Parameters
    ----------
    tList : (d,) array
        Time points for which HH is evaluated.
    Ilist : (d,) or (P, d) array
        Total input current for all t in tlist.

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
    _max = np.max(Ilist)
    _min = np.min(Ilist)
    ti, tf = tList[0], tList[-1]
    fig, ax = plt.subplots(figsize=(6,5),
              subplot_kw=dict(xlim=(ti-0.5, tf+0.5)
    , ylim=(_min-0.05,_max+0.05), xlabel='Time, in ms'
    , ylabel=r'Input Current $I_{\rm ext}~\left( \frac{\mu A}{cm^2}\right)$'))
    
    if len(Ilist.shape) == 2:
        for _i in range(Ilist.shape[0]):
            if _i == 0:  ax.plot(tList, Ilist[_i], color='darkslategrey', 
                                 lw=3, alpha=0.3)
            else:        ax.plot(tList, Ilist[_i])
    elif len(Ilist.shape) == 1:
        ax.plot(tList, Ilist, color='k', lw=2, alpha=0.3)
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax

def plotChannelAsymptotes(Vspace, n_inf):
    """
    Plots the asymptotic values of (m, n) as t approaches infinity, given
    a list of resting voltages Vspace.

    Parameters
    ----------
    Phispace : (numpoints,) array
        Range of voltage values from Vmin to Vmax.
    n_inf : (numpoints,) array
        Corresponding asymptotic values of n channel.
    
    See Also
    --------
    channelAsymptotes(Vmin, Vmax, numpoints) -> (Phispace, m_inf, n_inf) 

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
    fig, ax = plt.subplots(figsize=(6,4),
                           subplot_kw=dict(xlim=(Vspace[0]-0.5, Vspace[-1]+0.5)
                                         , ylim=(-0.05,1.05)
                                         , xlabel='Voltage, in mV'
                                         , ylabel='Gating channel'))
    ax.plot(Vspace, n_inf, color='goldenrod', lw=2, label='n channel')
    plt.legend(fontsize=12, loc=0)
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax

def plotTimeConstants(Phispace, n_tau):
    """
    Solves the time constants of n channel at steady-state given
    a list of resting voltages Phispace.

    Parameters
    ----------
    Phispace : (numpoints,) array
        Range of voltage values from Vmin to Vmax.
    n_tau : (numpoints,) array
        Corresponding time constant values of n channel.

    See Also
    --------
    timeConstants(Vmin, Vmax, numpoints) -> (Phispace, n_tau)    

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
    _max = np.max(np.ceil(n_tau))
    _min = np.min(np.floor(n_tau))
    fig, ax = plt.subplots(figsize=(6,4),
              subplot_kw=dict(xlim=(Phispace[0]-0.5, Phispace[-1]+0.5)
                            , ylim=(_min-.05,_max+.05)
                            , xlabel='Voltage, in mV'
                            , ylabel=r'Time constant, $\tau$'))
    
    ax.plot(Phispace, n_tau, color='goldenrod', lw=2, label='n channel')
    plt.legend(fontsize=12, loc=0)
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax

def plotISI(Ispace, ISIspace):
    """
    Plots the firing rate ISIspace given an array of input current Ispace.

    Parameters
    ----------
    Ispace : array
        Input current in uA/cm^2.
    ISIspace : array
        Firing rate in spikes/ms.
    
    Ispace and ISIspace must be of same length.

    See Also
    --------
    firingRate(V, t, Vthresh) -> ISI    

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
    _max = np.max(ISIspace)
    _min = np.min(ISIspace)
    fig, ax = plt.subplots(figsize=(6,6),
              subplot_kw=dict(xlim=(Ispace[0]-0.5, Ispace[-1]+0.5)
    , ylim=(_min-.005,_max+.005)
    , xlabel=r'Input current, $I_{\rm ext}~\left( \frac{\mu A}{cm^2}\right)$'
    , ylabel=r'Firing rate, $r$ (spikes/ms)'))
    
    ax.scatter(Ispace, ISIspace, marker='o', facecolors='none', edgecolors='k')
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax

def plotVoltagePhaseSpace(soln):
    """
    Plots the limit cycle of voltage Phi against channel n.

    Parameters
    ----------
    soln : (2, d) or (2, d, P) array
        where d = (tf-ti)/dt, and P = L*L.
        Values of (Phi, n) for all t in tList.

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
    V, n = soln
    _max = np.max(np.ceil(V))
    _min = np.min(np.floor(V))
    fig, ax = plt.subplots(figsize=(6,6),
                           subplot_kw=dict(xlim=(-0.05,1.05)
                                         , ylim=(_min-1,_max+1)
                                         , xlabel='Channel'
                                         , ylabel='Voltage'))
    ax.plot(n, V, color='goldenrod', lw=2, label='V-n')
    plt.legend(fontsize=12, loc=1)
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax