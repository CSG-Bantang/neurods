#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 21:30:24 2024

@author: reinierramos
"""

from neurods import MorrisLecar as ML
import numpy as np
from matplotlib import pyplot as plt

#%% Membrane voltage and gating channel over time

#%%% Single ML with constant input 
soln, Ilist, t  = ML.solveML(system='single', solver='lsoda', I0=60)
figVoltA1 = ML.plotVoltage(soln, t)
figVoltA1[1].set_title('Single ML with constant input')

figChanA1 = ML.plotChannels(soln, t)
figChanA1[1].set_title('Single ML with constant input')

figInputA1 = ML.plotCurrent(t, Ilist)
figInputA1[1].set_title('Single constant input')

#%%% Single ML with sinusoid input 
soln, Ilist, t   = ML.solveML(system='single', solver='rk4', tf=200, Is=10, fs=4.905)
figVoltA2 = ML.plotVoltage(soln, t)
figVoltA2[1].set_title('Single ML with sinusoid input')

figChanA2 = ML.plotChannels(soln, t)
figChanA2[1].set_title('Single ML with sinusoid input')

figInputA2 = ML.plotCurrent(t, Ilist)
figInputA2[1].set_title('Single ML sinusoid input')

#%%% Noisy ML with constant input 
soln, Ilist, t  = ML.solveML(system='noisy', solver='euler', I0=50, In=10)
figVoltB = ML.plotVoltage(soln, t)
figVoltB[1].set_title('Noisy ML with constant input')

figVoltB = ML.plotChannels(soln, t)
figVoltB[1].set_title('Noisy ML with constant input')

figInputB = ML.plotCurrent(t, Ilist)
figInputB[1].set_title('Noisy constant input')

#%%% Coupled ML with constant input
soln, Ilist, t  = ML.solveML(system='coupled', solver='euler', I0=60, L=3, g=0.9)
figVoltC = ML.plotVoltage(soln, t)
figVoltC[1].set_title('Coupled ML with constant input')

figInputC = ML.plotCurrent(t, Ilist)
figInputC[1].set_title('Coupled constant input')

#%%% Noisy Coupled ML with bias and sinusoid input
soln, Ilist, t  = ML.solveML(system='noisy coupled', solver='euler', 
                              I0=50, Is=10, fs=4.905, In=10, L=3, g=0.9)
figVoltD = ML.plotVoltage(soln, t)
figVoltD[1].set_title('Noisy Coupled ML with bias and sinusoid input')

figInputD = ML.plotCurrent(t, Ilist)
figInputD[1].set_title('Noisy coupled, bias, sinusoid input')

plt.show()
plt.close()

#%%Asymptotic Gating Channels and Time Constant
Vmin = -100
Vmax = 100
steps = 400

#%%% Asymptotic values
V, n = ML.channelAsymptotes(Vmin, Vmax, steps)
asymptotes = ML.plotChannelAsymptotes(V, n)
asymptotes[1].set_title('Asymptotic Gating Channels')

#%%% Time constants
V, n = ML.timeConstants(Vmin, Vmax, steps)
timeConstants = ML.plotTimeConstants(V, n)
timeConstants[1].set_title('Time Constants of Channels')

#%% Firing Rate
ti = 0
tf = 500
dt = 0.25
Vthresh = -20

Imin = 0
Imax = 200
Isteps = 50
Ispace = np.linspace(Imin, Imax, Isteps)
ISIspace = np.zeros(len(Ispace))
for i, I in enumerate(Ispace):
    soln, Ilist, t   = ML.solveML(system='single', solver='lsoda', I0=I,
                            ti=ti, tf=tf,dt=dt)
    V, _, = soln
    ISIspace[i] = ML.firingRate(V, t)

figISI = ML.plotISI(Ispace, ISIspace)
figISI[1].set_title('ML Activation Function')

#%% Phase Space of Voltage vs Channels
soln, Ilist, t  = ML.solveML(system='single', solver='lsoda', 
                              Vrest=0, I0=60, tf=1000, dt=0.1)
figVPS = ML.plotVoltagePhaseSpace(soln)
figVPS[1].set_title('Limit Cycles of ML')