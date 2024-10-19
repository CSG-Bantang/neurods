#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:12:43 2024

@author: reinierramos
"""

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from matplotlib import colors as mplc
from matplotlib import animation, colors

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

def plotDiscreteDensity(dsoln, soln):
    """
    Plots the density of states over time.

    Parameters
    ----------
    dsoln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of Discrete NCA.
    soln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of corresponding Neuronal CA.
        
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
    solveNCAD -> dsoln
    NeuronalCA.solveNCA -> soln

    """
    duration, L, _ = dsoln.shape
    N = L*L
    densityF = np.sum(dsoln==F, axis=(1,2)) / N
    densityR = np.sum(dsoln==R, axis=(1,2)) / N
    densityQ = np.sum(dsoln==Q, axis=(1,2)) / N
    density  = np.mean(soln, axis=(1,2))
    fig, ax = plt.subplots(figsize=(6,5),
                           subplot_kw=dict(xlim=(-0.5, duration+0.5)
                                         , ylim=(-0.05, 1.05)
                                         , xlabel='Generation'
                                         , ylabel='Density of Cells'))
    ax.plot(range(duration), densityR, color='rebeccapurple', lw=1, 
            label='R', ls='--')
    ax.plot(range(duration), densityQ, color='black', ls='--', marker= 'd',
            markersize=7, markerfacecolor='None', label='Q', alpha=0.6)
    ax.plot(range(duration), densityF, color='sandybrown', marker= 'o', 
            markersize=10, markerfacecolor='w', label='F', lw=2)
    ax.plot(range(duration), density, color='darkgoldenrod', lw=3, marker= 'o', 
            markersize=7, markerfacecolor='w', label=r'$\langle a \rangle$')
    ax.locator_params(axis='both', tight=True, nbins=5)
    plt.legend(fontsize=12, loc=1)
    return fig, ax

def animateNCAD(dsoln, out='animNCAD.gif'):
    """
    Saves the spatiotemporal dynamics of Discrete NCA as GIF.

    Parameters
    ----------
    dsoln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of NCA.
    out : str, default is 'animNCAD.gif'
        Output file name of the GIF.
        Must end with '.gif'
    
    See Also
    --------
    solveNCAD -> dsoln
        
    """
    duration, L, _ = dsoln.shape
    resize = 200
    ims = [Image.fromarray(np.uint8(QFRcmap(dsoln[i,:,:]/2)*255)) 
           for i in range(duration)]
    ims = [im.convert('P', palette=Image.ADAPTIVE, colors=3) for im in ims]
    ims = [ImageOps.contain(im, (resize,resize)) for im in ims]
    ims[0].save(fp=out, format='gif', append_images=ims, save_all=True, 
                duration=duration, loop=0)

def animate3DNCAD(dsoln, out='anim3DNCAD.gif'):
    """
    Saves the spatiotemporal dynamics of Spherical Discrete NCA as GIF.
    Use when specifying system = 5, 6, 7, or 8.

    Parameters
    ----------
    dsoln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of NCA.
    out : str, default is 'anim3DNCAD.gif'
        Output file name of the GIF.
        Must end with '.gif'
    
    See Also
    --------
    solveNCAD -> dsoln
        
    """
    duration, L, _ = dsoln.shape
    fig = plt.figure(figsize=(7,7), layout='tight')
    ax = fig.add_subplot(projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    u, v = np.mgrid[0:np.pi:complex(0,L), 0:2*np.pi:complex(0,L)]
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    cmap=QFRcmap
    norm=colors.Normalize(vmin = 0,
                          vmax = 2, clip = False)
    im = ax.plot_surface(x,y,z, cmap=cmap, rstride=1, cstride=1,
                         linewidth=0.01, facecolors=cmap(norm(dsoln[0,:,:])))
    ax.axis('off')
    ax.set_xlim(np.array([-1,1])*.6)
    ax.set_ylim(np.array([-1,1])*.65)
    ax.set_zlim(np.array([-1,1])*.6)
    ax.set_aspect('equal')
    anim = animation.FuncAnimation(fig, update, frames=range(duration),
                                   fargs=(dsoln, im, cmap, norm, L),
                                   interval=100, blit=True)
    anim.save(out, dpi=200)

def update(t:int, dsoln:list[float], im, cmap, norm, L:int):
    """
    Update function used for generating artists for FuncAnimation.

    """
    im.set_facecolors(cmap(norm(dsoln[t,:,:]))[:-1, :-1].reshape(L-1*L-1, 4))
    return im,