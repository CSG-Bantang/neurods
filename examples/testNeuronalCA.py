#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:25:33 2024

@author: reinierramos
"""

from neurods import NeuronalCA as NCA
import os

# %% Neuronal Cellular Automata

gifpath = './NeuronalCA-gifs'
os.makedirs(gifpath, exist_ok=True)

#%%% Linear Activation
a0 = 0.0
a1 = 0.2
a2 = 0.7
nl = 1

ain, aout = NCA.ncaReturnMap(a0, a1, a2, nl)
linearMap = NCA.plotActivation(ain, aout)
linearMap[1].set_title('Linear Activation')

soln = NCA.solveNCA(a0=a0, a1=a1, a2=a2, nl=nl, duration=50)
NCA.animateNCA(soln, out=f'{gifpath}/linear.gif')

linear = NCA.plotDensity(soln)
linear[1].set_title('Neuronal CA with linear activation')

#%%% Nonlinear Activation
a0 = 0.0
a1 = 1.0
a2 = 0.9
nl = 2.0

ain, aout = NCA.ncaReturnMap(a0, a1, a2, nl)
nonlinearMap = NCA.plotActivation(ain, aout)
nonlinearMap[1].set_title('Nonlinear Activation')

soln = NCA.solveNCA(a0=a0, a1=a1, a2=a2, nl=nl)
NCA.animateNCA(soln, out=f'{gifpath}/nonlinear.gif')
nonlinear = NCA.plotDensity(soln)
nonlinear[1].set_title('Neuronal CA with nonlinear activation')

#%%% Epileptic Activation
a0 = 0.8
a1 = 0.0
a2 = 0.9
nl = 1.0

ain, aout = NCA.ncaReturnMap(a0, a1, a2, nl)
epilepticMap = NCA.plotActivation(ain, aout)
epilepticMap[1].set_title('Epileptic Activation')

soln = NCA.solveNCA(a0=a0, a1=a1, a2=a2, nl=nl)
NCA.animateNCA(soln, out=f'{gifpath}/epileptic.gif')
epileptic = NCA.plotDensity(soln)
epileptic[1].set_title('Neuronal CA with epileptic activation')

#%%% Nonlinear with spherical boundary
a0 = 0.2
a1 = 0.7
a2 = 0.9
nl = 4.0

ain, aout = NCA.ncaReturnMap(a0, a1, a2, nl)
nonlinearMap = NCA.plotActivation(ain, aout)
nonlinearMap[1].set_title('Nonlinear Activation used for spherical NCA')

soln = NCA.solveNCA(a0=a0, a1=a1, a2=a2, nl=nl, system=8)
NCA.animateNCA(soln, out=f'{gifpath}/spherical-2D.gif')
NCA.animate3DNCA(soln,  out=f'{gifpath}/spherical-3D.gif')
sphere = NCA.plotDensity(soln)
sphere[1].set_title('Neuronal CA with spherical boundary')

#%% Neuronal CA with initial beta distribution

a0 = 0.2
a1 = 0.7
a2 = 0.9
nl = 4.0

ain, aout = NCA.ncaReturnMap(a0, a1, a2, nl)
nonlinearMap = NCA.plotActivation(ain, aout)
nonlinearMap[1].set_title('Nonlinear Activation used for initial beta')

#%%% beta(a,b) initial
a = 5
b = 3
soln=NCA.solveNCA(a0=a0, a1=a1, a2=a2, nl=nl, a=a, b=b)
NCA.animateNCA(soln, out=f'{gifpath}/beta-ab.gif')
beta = NCA.plotDensity(soln)
beta[1].set_title(fr"Neuronal CA with $\beta(a={a}, b={b})$ initial")

#%%% beta(mu,nu) initial
mu = 0.8
nu = 4
soln = NCA.solveNCA(a0=a0, a1=a1, a2=a2, nl=nl, mu=mu, nu=nu)
NCA.animateNCA(soln, out=f'{gifpath}/beta-mn.gif')
beta = NCA.plotDensity(soln)
beta[1].set_title(fr"Neuronal CA with $\beta(\mu={mu},\nu={nu})$ initial")

#%% Nonlinear with spike defective neurons
a0 = 0.2
a1 = 0.7
a2 = 0.9
nl = 3.5

ain, aout = NCA.ncaReturnMap(a0, a1, a2, nl)
nonlinearMap = NCA.plotActivation(ain, aout)
nonlinearMap[1].set_title('Nonlinear Activation used for NCA with defect')

defect = 0.2
soln, defects = NCA.solveNCA(a0=a0, a1=a1, a2=a2, nl=nl, defect=defect)
NCA.animateNCA(soln, out=f'{gifpath}/with-defect.gif')
defective = NCA.plotDensity(soln)
defective[1].set_title(f'Neuronal CA with {int(defect*100)}% defect')

