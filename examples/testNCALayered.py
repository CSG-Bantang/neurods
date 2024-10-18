#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:47:51 2024

@author: reinierramos
"""

from neurods import NCALayered as NCAL
import os

a0 = 0.0
a1 = 0.8
a2 = 0.9
nl = 2.0

ain, aout = NCAL.ncaReturnMap(a0, a1, a2, nl)
nonlinearMap = NCAL.plotActivation(ain, aout)
nonlinearMap[1].set_title('Nonlinear Activation')

layers = 6
L = 6
soln = NCAL.solveNCAL(a0=a0, a1=a1, a2=a2, nl=nl, L = L, layers=layers)

layered = NCAL.plotDensity(soln)
layered[1].set_title(f'Neuronal CA with {layers} layers')


gifpath = './NeuronalCA-gifs'
os.makedirs(gifpath, exist_ok=True)

NCAL.animateNCAL(soln,   out=f'{gifpath}/{layers}-layers-2D.gif')
NCAL.animate3DNCAL(soln, out=f'{gifpath}/{layers}-layers-3D.gif')