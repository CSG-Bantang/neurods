#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:25:10 2024

@author: reinierramos
"""

from neurods import HodgkinHuxley as HH
import numpy as np
from matplotlib import pyplot as plt

#%% Membrane voltage and gating channel over time

#%%% Single HH with constant input 
soln, Ilist, t  = HH.solveHH(system='single', solver='lsoda', I0=2.5)
figVoltA1 = HH.plotVoltage(soln, t)
figVoltA1[1].set_title('Single HH with constant input')

figChanA1 = HH.plotChannels(soln, t)
figChanA1[1].set_title('Single HH with constant input')

figInputA1 = HH.plotCurrent(t, Ilist)
figInputA1[1].set_title('Single constant input')

#%%% Single HH with sinusoid input 
soln, Ilist, t   = HH.solveHH(system='single', solver='rk4', tf=200,Is=10, fs=4.905)
figVoltA2 = HH.plotVoltage(soln, t)
figVoltA2[1].set_title('Single HH with sinusoid input')

figChanA2 = HH.plotChannels(soln, t)
figChanA2[1].set_title('Single HH with sinusoid input')

figInputA2 = HH.plotCurrent(t, Ilist)
figInputA2[1].set_title('Single HH sinusoid input')

#%%% Noisy HH with constant input 
soln, Ilist, t  = HH.solveHH(system='noisy', solver='euler', I0=2.5, In=60)
figVoltB = HH.plotVoltage(soln, t)
figVoltB[1].set_title('Noisy HH with constant input')

figVoltB = HH.plotChannels(soln, t)
figVoltB[1].set_title('Noisy HH with constant input')

figInputB = HH.plotCurrent(t, Ilist)
figInputB[1].set_title('Noisy constant input')

#%%% Coupled HH with constant input
soln, Ilist, t  = HH.solveHH(system='coupled', solver='euler', I0=10, L=3, g=0.1)
figVoltC = HH.plotVoltage(soln, t)
figVoltC[1].set_title('Coupled HH with constant input')

figInputC = HH.plotCurrent(t, Ilist)
figInputC[1].set_title('Coupled constant input')

#%%% Noisy Coupled HH with bias and sinusoid input
soln, Ilist, t  = HH.solveHH(system='noisy coupled', solver='euler', 
                              I0=2.5, Is=10, fs=4.905, In=60, L=3, g=0.1)
figVoltD = HH.plotVoltage(soln, t)
figVoltD[1].set_title('Noisy Coupled HH with bias and sinusoid input')

figInputD = HH.plotCurrent(t, Ilist)
figInputD[1].set_title('Noisy coupled, bias, sinusoid input')

plt.show()
plt.close()

#%%Asymptotic Gating Channels and Time Constant
Vmin = -100
Vmax = 100
steps = 400

#%%% Asymptotic values
V, m, h, n = HH.channelAsymptotes(Vmin, Vmax, steps)
asymptotes = HH.plotChannelAsymptotes(V, m, h, n)
asymptotes[1].set_title('Asymptotic Gating Channels')

#%%% Time constants
V, m, h, n = HH.timeConstants(Vmin, Vmax, steps)
timeConstants = HH.plotTimeConstants(V, m, h, n)
timeConstants[1].set_title('Time Constants of Channels')

#%% Firing Rate
ti = 0
tf = 500
dt = 0.25
Vthresh = 19

Imin = 0
Imax = 200
Isteps = 50
Ispace = np.linspace(Imin, Imax, Isteps)
ISIspace = np.zeros(len(Ispace))
for i, I in enumerate(Ispace):
    soln, Ilist, t   = HH.solveHH(system='single', solver='lsoda', I0=I,
                            ti=ti, tf=tf,dt=dt)
    V, _, _, _ = soln
    ISIspace[i] = HH.firingRate(V, t)

figISI = HH.plotISI(Ispace, ISIspace)
figISI[1].set_title('HH Activation Function')

#%% Phase Space of Voltage vs Channels
soln, Ilist, t  = HH.solveHH(system='single', solver='lsoda', 
                              Vrest=-70, I0=2.5, tf=50, dt=0.025)
figVPS = HH.plotVoltagePhaseSpace(soln)
figVPS[1].set_title('Limit Cycles of HH Part 1')
figCPS = HH.plotChannelPhaseSpace(soln)
figCPS[1].set_title('Limit Cycles of HH Part 2')