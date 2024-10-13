#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:52:12 2024

@author: reinierramos
"""

from matplotlib import pyplot as plt
import numpy as np

import HodgkinHuxley as HH

#%%% Membrane voltage and gating channel over time
#%%% Single HH with constant input 
soln, t   = HH.solveHH(system='single', solver='lsoda', I0=2.5)
figVoltA1 = HH.plotVoltage(soln, t)
figVoltA1[1].set_title('Single HH with constant input')
figChanA1 = HH.plotChannels(soln, t)
figChanA1[1].set_title('Single HH with constant input')
#%%% Single HH with sinusoid input 
soln, t   = HH.solveHH(system='single', solver='rk4', Is=10, fs=4.905)
figVoltA2 = HH.plotVoltage(soln, t)
figVoltA2[1].set_title('Single HH with sinusoid input')
figChanA2 = HH.plotChannels(soln, t)
figChanA2[1].set_title('Single HH with sinusoid input')
#%%% Noisy HH with constant input 
soln, t  = HH.solveHH(system='noisy', solver='euler', In=60)
figVoltB = HH.plotVoltage(soln, t)
figVoltB[1].set_title('Noisy HH with constant input')
figVoltB = HH.plotChannels(soln, t)
figVoltB[1].set_title('Noisy HH with constant input')
#%%% Coupled HH with constant input
soln, t  = HH.solveHH(system='coupled', solver='euler', I0=10, L=3, g=0.1)
figVoltC = HH.plotVoltage(soln, t)
figVoltC[1].set_title('Coupled HH with constant input')
#%%% Noisy Coupled HH with bias and sinusoid input
soln, t  = HH.solveHH(system='noisy coupled', solver='euler', I0=2.5, Is=10, fs=4.905, In=60, L=3, g=0.1)
figVoltD = HH.plotVoltage(soln, t)
figVoltD[1].set_title('Noisy Coupled HH with bias and sinusoid input')
plt.show()
plt.close()

#%%Asymptotic Gating Channels and Time Constant
Vmin = -100
Vmax = 100
steps = 400
#%%% Asymptotic values
V, m, h, n = HH.asymptoticChannels(Vmin, Vmax, steps)
asymptotes = HH.plotAsympChannels(V, m, h, n)
asymptotes[1].set_title('Asymptotic Gating Channels')
#%%% Time constants
V, m, h, n = HH.timeConstants(Vmin, Vmax, steps)
timeConstants = HH.plotTimeConstants(V, m, h, n)
timeConstants[1].set_title('Time Constants of Channels')

#%% Firing Rate
ti = 0
tf = 1000
dt = 0.25
Vthresh = 19

Imin = 0
Imax = 200
Isteps = 100
Ispace = np.linspace(Imin, Imax, Isteps)
ISIspace = np.zeros(len(Ispace))
for i, I in enumerate(Ispace):
    soln, t   = HH.solveHH(system='single', solver='lsoda', I0=I,
                           ti=ti, tf=tf,dt=dt)
    V, _, _, _ = soln
    ISIspace[i] = HH.firingRate(V, t)

figISI = HH.plotISI(Ispace, ISIspace)
figISI[1].set_title('HH Activation Function')








