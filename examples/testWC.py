#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:27:21 2024

@author: reinierramos
"""

from neurods import WilsonCowan as WC

soln, I_elist, I_ilist, t = WC.solveWC(solver='lsoda', E0=0.1, I0=0.05, 
                                       dt=0.025)
density = WC.plotDensity(soln, t)
density[1].set_title('Single WC, no input current')

noinput = WC.plotCurrent(t, I_elist, I_ilist)
noinput[1].set_title('Single WC, no input current')

trajectory = WC.plotDensityPhaseSpace(soln)
trajectory[1].set_title('Single WC, no input current')

soln, I_elist, I_ilist, t = WC.solveWC(solver='lsoda', E0=0.1, I0=0.05, 
                                       dt=0.025, fs=40, Is_e=10, Is_i=4)
density = WC.plotDensity(soln, t)
density[1].set_title('Single WC, with sinusoidal current')

sineinput = WC.plotCurrent(t, I_elist, I_ilist)
sineinput[1].set_title('Single WC, sinusoidal current')

trajectory = WC.plotDensityPhaseSpace(soln)
trajectory[1].set_title('Single WC, sinusoidal current')

