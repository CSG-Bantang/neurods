#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:59:04 2024

@author: reinierramos
"""

import numpy as np
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

def plotVoltage(soln, tList):
    """
    Plotter function for membrane voltage V(t, in ms) in mV.

    Parameters
    ----------
    soln : 2D or 3D ndarray
        Values of V, m, h, n for al `t` in `tList`.
    tList : 1D ndarray
        Time points for which HH is evaluated.

    Returns
    -------
    fig : matplotlib.figure.Figure object
        Figure instance for which V(t) is plotted.
        Has all the attributes of matplotlib.figure.Figure
    ax : matplotlib.axes._axes.Axes object
        Axes instance for which V(t) is plotted.
        Has all the attributes of matplotlib.axes._axes.Axes
    
    `fig` and `ax` are the same as if `fig, ax = plt.subplots()` is called.

    """
    ti, tf = tList[0], tList[-1]
    fig, ax = plt.subplots(figsize=(6,5),
                           subplot_kw=dict(xlim=(ti-0.5, tf+0.5)
                                         , ylim=(-20,120)
                                         , xlabel='Time, in ms'
                                         , ylabel='Voltage, in mV'))
    V, _, _, _ = soln
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
    Plotter function for activation probability of channels 
    m, h, n over time t in ms.

    Parameters
    ----------
    soln : 2D or 3D ndarray
        Values of V, m, h, n for al `t` in `tList`.
    tList : 1D ndarray
        Time points for which HH is evaluated.

    Returns
    -------
    fig : matplotlib.figure.Figure object
        Figure instance for which V(t) is plotted.
        Has all the attributes of matplotlib.figure.Figure
    ax : matplotlib.axes._axes.Axes object
        Axes instance for which V(t) is plotted.
        Has all the attributes of matplotlib.axes._axes.Axes
    
    `fig` and `ax` are the same as if `fig, ax = plt.subplots()` is called.

    """
    ti, tf = tList[0], tList[-1]
    fig, ax = plt.subplots(figsize=(6,4),
                           subplot_kw=dict(xlim=(ti-0.5, tf+0.5)
                                         , ylim=(-0.05,1.05)
                                         , xlabel='Time, in ms'
                                         , ylabel='Gating variable'))
    _, m, h, n = soln    
    ax.plot(tList, m, color='darkgreen', lw=2, label='Na activation')
    ax.plot(tList, h, color='turquoise', lw=2, label='Na inactivation')
    ax.plot(tList, n, color='goldenrod', lw=2, label='K activation')
    plt.legend(fontsize=12, loc=1)
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax

def plotAsympChannels(Vspace, asymp_m, asymp_h, asymp_n):
    fig, ax = plt.subplots(figsize=(6,4),
                           subplot_kw=dict(xlim=(Vspace[0]-0.5, Vspace[-1]+0.5)
                                         , ylim=(-0.05,1.05)
                                         , xlabel='Time, in ms'
                                         , ylabel='Gating channel'))
    ax.plot(Vspace, asymp_m, color='darkgreen', lw=2, label='Na activation')
    ax.plot(Vspace, asymp_h, color='turquoise', lw=2, label='Na inactivation')
    ax.plot(Vspace, asymp_n, color='goldenrod', lw=2, label='K activation')
    plt.legend(fontsize=12, loc=0)
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax

def plotTimeConstants(Vspace, tau_m, tau_h, tau_n):
    _max = np.max(np.ceil(np.array([tau_m, tau_h, tau_n])))
    _min = np.min(np.floor(np.array([tau_m, tau_h, tau_n])))
    fig, ax = plt.subplots(figsize=(6,4),
                           subplot_kw=dict(xlim=(Vspace[0]-0.5, Vspace[-1]+0.5)
                                         , ylim=(_min-.05,_max+.05)
                                         , xlabel='Time, in ms'
                                         , ylabel='Time constant'))
    ax.plot(Vspace, tau_m, color='darkgreen', lw=2, label='Na activation')
    ax.plot(Vspace, tau_h, color='turquoise', lw=2, label='Na inactivation')
    ax.plot(Vspace, tau_n, color='goldenrod', lw=2, label='K activation')
    plt.legend(fontsize=12, loc=0)
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax

def plotISI(Ispace, ISIspace):
    _max = np.max(ISIspace)
    _min = np.min(ISIspace)
    fig, ax = plt.subplots(figsize=(6,6),
                           subplot_kw=dict(xlim=(Ispace[0]-0.5, Ispace[-1]+0.5)
                                         , ylim=(_min-.005,_max+.005)
                                         , xlabel=r'Input current, $I_{\rm ext}~\left( \frac{\mu A}{cm^2}\right)$'
                                         , ylabel=r'Firing rate, $r$ (spikes/ms)'))
    ax.scatter(Ispace, ISIspace, s=70, edgecolors='darkslategrey', facecolors='None')
    ax.locator_params(axis='both', tight=True, nbins=5)
    return fig, ax







