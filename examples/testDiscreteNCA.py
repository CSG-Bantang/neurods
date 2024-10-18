#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:36:19 2024

@author: reinierramos
"""

from neurods import NCADiscrete as NCAD
from neurods import NeuronalCA as NCA

import os

#%% Nonlinear Activation
a0 = 0.2
a1 = 0.7
a2 = 0.9
nl = 4.0
soln = NCA.solveNCA(system=7, a0=a0, a1=a1, a2=a2, nl=nl)

ain, aout = NCA.ncaReturnMap(a0, a1, a2, nl)
nonlinearMap = NCA.plotActivation(ain, aout)
nonlinearMap[1].set_title('Nonlinear Activation')

dsoln = NCAD.solveNCAD(soln)

discrete = NCAD.plotDiscreteDensity(dsoln, soln)
discrete[1].set_title('Discrete Neuronal CA with nonlinear activation')

gifpath = './NCADiscrete-gifs'
os.makedirs(gifpath, exist_ok=True)
NCAD.animateNCAD(dsoln,   out=f'{gifpath}/nonlinear-2D.gif')
NCAD.animate3DNCAD(dsoln, out=f'{gifpath}/nonlinear-3D.gif')
