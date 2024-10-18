#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:34:12 2024

@author: reinierramos
"""

from neurods import NCADiscreteLayered as NCADL
from neurods import NCALayered as NCAL

import os

#%% Nonlinear Activation
a0 = 0.0
a1 = 0.8
a2 = 0.9
nl = 2.0

ain, aout = NCAL.ncaReturnMap(a0, a1, a2, nl)
nonlinearMap = NCAL.plotActivation(ain, aout)
nonlinearMap[1].set_title('Nonlinear Activation')

layers = 6
L = 6
soln = NCAL.solveNCAL(system=6, layers=layers, L = L, a0=a0, a1=a1, a2=a2, nl=nl)

dsoln = NCADL.solveNCADL(soln)
disclay = NCADL.plotDiscreteDensity(dsoln, soln)
disclay[1].set_title('Discrete Layered NCA with nonlinear activation')

gifpath = './NCADiscrete-gifs'
os.makedirs(gifpath, exist_ok=True)
NCADL.animateNCADL(dsoln,   out=f'{gifpath}/{layers}-layers-2D.gif')
NCADL.animate3DNCADL(dsoln, out=f'{gifpath}/{layers}-layers-3D.gif')
