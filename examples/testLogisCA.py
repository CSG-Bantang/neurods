#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:54:35 2024

@author: reinierramos
"""

from neurods import LogisticMapCA as LMCA

import numpy as np
import os
from matplotlib import pyplot as plt

#%% Steady State x(t)
#%%% Individual Steady State x(t)
x0 = 0.3
r  = 4
ti = 0
tf = 25
dt = 1
x, t = LMCA.solveLM(r, x0, ti=0, tf=25, dt=1)

figSteadyState = LMCA.plotXvsT(x,t)
figSteadyState[1].set_title(f'Steady-state of LM system, $r={r}$, $x_0={x0}$')
plt.show()
plt.close()

#%%% Steady State x(t) varying r
x0 = 0.3
rList = range(5)
ti = 0
tf = 25
dt = 1
fig, ax = plt.subplots(figsize=(6,5),
                        subplot_kw=dict(xlim=(ti-0.5, tf+0.5)
                                      , ylim=(-0.05,1.05)
                                      , xlabel='Timestep, t'
                                      , ylabel='Steady-state, x(t)'))
for i, r in enumerate(rList):
    x, t = LMCA.solveLM(r, x0, ti=0, tf=25, dt=1)
    ax.plot(t, x, lw=1, marker='d', markersize=5, markerfacecolor='white', label=f'r = {r}')
ax.set_title(fr'Steady state with $x_0 = {x0}$')
ax.locator_params(axis='both', tight=True, nbins=5)
plt.legend(fontsize=12, loc=1)

#%%% Steady State x(t) varying x0
x0List = np.asarray([0.25, 0.5, 0.9])
r = 4
ti = 0
tf = 25
dt = 1
fig, ax = plt.subplots(figsize=(6,5),
                        subplot_kw=dict(xlim=(ti-0.5, tf+0.5)
                                      , ylim=(-0.05,1.05)
                                      , xlabel='Timestep, t'
                                      , ylabel='Steady-state, x(t)'))
for i, x0 in enumerate(x0List):
    x, t = LMCA.solveLM(r, x0, ti=0, tf=25, dt=1)
    ax.plot(t, x, lw=1, marker='d', markersize=5, markerfacecolor='white', label=fr'$x_0 = {x0}$')
ax.set_title(fr'Steady state with $r = {r}$')
ax.locator_params(axis='both', tight=True, nbins=5)
plt.legend(fontsize=12, loc=1)

#%% Return Map
#%%% Individual Return Map
r = 2
x, y = LMCA.logisticReturnMap(r)
figLMap = LMCA.plotReturnMap(x, y)
figLMap[1].set_title(f'Return map of LM system, $r={r}$')
plt.show()
plt.close()


#%%% Return Map for varying r
rList = range(5)
fig, ax = plt.subplots(figsize=(6,5),
                        subplot_kw=dict(xlim=(-0.05,1.05)
                                      , ylim=(-0.05,1.05)
                                      , xlabel='Previous Steady-State, $x_{t}$'
                                      , ylabel='Next Steady-State, $x_{t+1}$'))
for i, r in enumerate(rList):
    x, y = LMCA.logisticReturnMap(r)
    ax.plot(x, y, lw=4, label=f'r = {r}')
ax.set_title('Return map for varying r')
ax.locator_params(axis='both', tight=True, nbins=5)
plt.legend(fontsize=12, loc=1)


#%% Logistic CA
r=4

x, y = LMCA.logisticReturnMap(r)
returnMap = LMCA.plotReturnMap(x,y)
returnMap[1].set_title('Return map used in Logistic CA')

gifpath = './LogisCA-gifs'
os.makedirs(gifpath, exist_ok=True)


#%%% LM CA with uniform initial
soln = LMCA.solveLMCA(r=r)

LMCA.animateLMCA(soln, out=f'{gifpath}/uniform.gif')
uniform = LMCA.plotDensity(soln)
uniform[1].set_title('Logistic CA with uniform initial')

#%%% LM CA with spherical lattice boundary
soln = LMCA.solveLMCA(system=8, r=r)

LMCA.animateLMCA(soln, out=f'{gifpath}/spherical-2D.gif')
LMCA.animate3DLMCA(soln,  out=f'{gifpath}/spherical-3D.gif')
spherical = LMCA.plotDensity(soln)
spherical[1].set_title('Logistic CA with spherical boundary')

#%%% Logistic CA with beta(a,b) initial
a = 5
b = 3
soln = LMCA.solveLMCA(r=r, a=a, b=b)
LMCA.animateLMCA(soln, out=f'{gifpath}/beta-ab.gif')
beta = LMCA.plotDensity(soln)
beta[1].set_title(fr"Logistic CA with $\beta(a={a}, b={b})$ initial")

#%%% Logistic CA with beta(mu,nu) initial
mu = 0.8
nu = 4
soln = LMCA.solveLMCA(r=r, mu=mu, nu=nu)
LMCA.animateLMCA(soln, out=f'{gifpath}/beta-munu.gif')
beta = LMCA.plotDensity(soln)
beta[1].set_title(fr"Logistic CA with $\beta(\mu={mu},\nu={nu})$ initial")


