#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:41:15 2024

@author: reinierramos
"""

"""
Author's Note:
If running in Spyder or IPython, make sure the graphics backend is set to 
either "Automatic" or "Qt5" so that animations work. This is necessary for
mayavi to run and generate plots.

See more here: 
    https://docs.enthought.com/mayavi/mayavi/mlab_running_scripts.html
"""

from neurods import NeuronalSV as NSV
import os

#%%% Linear Activation
a0 = 0.0
a1 = 0.8
a2 = 0.9
nl = 1

soln, edges, xyz = NSV.solveNSV(L=16, a0=a0, a1=a1, a2=a2, nl=nl)

gifpath = './NeuronalSV-gifs'
os.makedirs(gifpath, exist_ok=True)

print("Animating...")
NSV.animateNSV(soln, edges, xyz, out=f'{gifpath}/random.gif')


### Switching backends back to inline
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline', force=True)

ain, aout = NSV.ncaReturnMap(a0, a1, a2, nl)
linearMap = NSV.plotActivation(ain, aout)
linearMap[1].set_title('Linear Activation')

linear = NSV.plotDensity(soln)
linear[1].set_title('Neuronal SV with linear activation')

