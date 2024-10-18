#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:04:48 2024

@author: reinierramos
"""

import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from matplotlib import animation, colors
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

def plotDiscreteDensity(dsoln, soln):
    """
    Plots the density of states over time.

    Parameters
    ----------
    dsoln : (duration, layers, L, L) array
        Snapshots of the spatiotemporal dynamics of Discrete NCA.
    soln : (duration, layers, L, L) array
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
    solveNCADL -> dsoln
    NCALayered.solveNCAL -> soln

    """
    duration, layers, L, _ = dsoln.shape
    N = L*L*layers
    densityQ = np.sum(dsoln==Q, axis=(1,2,3)) / N
    densityF = np.sum(dsoln==F, axis=(1,2,3)) / N
    densityR = np.sum(dsoln==R, axis=(1,2,3)) / N
    densLayF = np.sum(dsoln==F, axis=(2,3)) / N
    densLayR = np.sum(dsoln==R, axis=(2,3)) / N
    density = np.mean(soln, axis=(1,2,3))
    densLay = np.mean(soln, axis=(2,3))
    cmap = plt.get_cmap('winter')
    layerColors = [cmap(i) for i in np.linspace(0, 1, layers)]
    fig, ax = plt.subplots(figsize=(6,5),
                           subplot_kw=dict(xlim=(-0.5, duration+0.5)
                                         , ylim=(-0.05, 1.05)
                                         , xlabel='Generation'
                                         , ylabel='Density of Cells'))
    for i, layerColor in enumerate(layerColors, start=1):
        ax.plot(range(duration), densLay[:,i-1], color=layerColor, lw=1, 
                label=fr' $\langle a \rangle$ layer {i}', alpha=0.6)
        ax.plot(range(duration), densLayF[:,i-1], color=layerColor, lw=1, alpha=0.6)
        ax.plot(range(duration), densLayR[:,i-1], color=layerColor, lw=1, alpha=0.6)
    ax.plot(range(duration), densityR, color='rebeccapurple', lw=3, 
            label='R all layers')
    ax.plot(range(duration), densityQ, color='black', ls='--', marker= 'd',
            markersize=7, markerfacecolor='None', label='Q all layers', alpha=0.6)
    ax.plot(range(duration), density, color='darkgoldenrod', lw=3, marker= 'o', 
            markersize=7, markerfacecolor='w', 
            label=r'$\langle a \rangle$ all layers')
    ax.plot(range(duration), densityF, color='sandybrown', lw=3, marker= 'o', 
            markersize=7, markerfacecolor='w', label='F all layers')
    ax.locator_params(axis='both', tight=True, nbins=5)
    plt.legend(fontsize=8, loc=1)
    return fig, ax

def animateNCADL(dsoln, out='animNCADL.gif'):
    """
    Saves the spatiotemporal dynamics of Discrete Layered NCA as GIF.

    Parameters
    ----------
    dsoln : (duration, L, L) array
        Snapshots of the spatiotemporal dynamics of Disceret Layered NCA.
    out : str, default is 'animNCADL.gif'
        Output file name of the GIF.
        Must end with '.gif'
    
    See Also
    --------
    solveNCADL -> dsoln
        
    """
    duration, layers, L, _ = dsoln.shape
    resize = 200
    soln = np.hstack(tuple([dsoln[:,layer,:,:] for layer in range(layers)]))
    ims = [Image.fromarray(np.uint8(QFRcmap(soln[i,:,:])*255)) for i in range(duration)]
    ims = [im.convert('P', palette=Image.ADAPTIVE, colors=100) for im in ims]
    ims = [ImageOps.contain(im, (resize,resize)) for im in ims]
    ims[0].save(fp=out, format='gif', append_images=ims, save_all=True, 
                duration=duration, loop=0)
    
def animate3DNCADL(dsoln, out='anim3DNCADL.gif'):
    """
    Saves the spatiotemporal dynamics of 3D Discrete Layered NCA as GIF.

    Parameters
    ----------
    dsoln : (duration, layers, L, L) array
        Snapshots of the spatiotemporal dynamics of Discrete Layered NCA.
    out : str, default is 'anim3DNCADL.gif'
        Output file name of the GIF.
        Must end with '.gif'
    
    See Also
    --------
    solveNCADL -> dsoln
    
    """
    duration, layers, L, _ = dsoln.shape
    cmap=QFRcmap
    x,y,z = np.indices((layers+1,L+1,L+1))
    filled = np.ones((layers,L,L))
    norm=colors.Normalize(vmin = 0, vmax = 2, clip = False)
    facecolors = cmap(norm(dsoln[0,:,:,:]))
    alpha = 0.5
    facecolors[:,:,:,3] = alpha
    fig = plt.figure(figsize=(7,7), layout='tight')
    ax = fig.add_subplot(projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    im = ax.voxels(x,y,z, filled, facecolors=facecolors, edgecolor='k',
              linewidth=0.01)
    
    ax.axis('off')
    ax.set_aspect('equal')
    anim = animation.FuncAnimation(fig, update, frames=range(duration),
                                   fargs=(dsoln, ax, cmap, norm, x,y,z,filled, alpha),
                                   interval=100, blit=True)
    anim.save(out, dpi=200)
    
def update(t:int, dsoln:list[float], ax, cmap, norm, x,y,z,filled, alpha):
    """
    Update function used for generating artists for FuncAnimation.

    """
    facecolors = cmap(norm(dsoln[t,:,:,:]))
    facecolors[:,:,:,3] = alpha
    im = ax.voxels(x,y,z, filled, facecolors=facecolors, edgecolor='k',
              linewidth=0.01)
    return list(im.values())