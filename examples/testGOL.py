#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:25:04 2024

@author: reinierramos
"""

from neurods import GameofLife as GOL
import os

#%% Game of Life Systems

gifpath = './GOL-gifs'
os.makedirs(gifpath, exist_ok=True)

#%%% GOL CA with uniform initial
soln=GOL.solveGOL(system=0, L=50, p=0.5, duration=40)
GOL.animateGOL(soln, out=f'{gifpath}/random.gif')
uniform = GOL.plotDensity(soln)
uniform[1].set_title('Density with initial uniform random')

#%%% GOL CA initialized as Beehive
soln=GOL.solveGOL(system=2, duration=30)
GOL.animateGOL(soln, out=f'{gifpath}/beehive.gif')
beehive = GOL.plotDensity(soln)
beehive[1].set_title('Density with initial beehive')

#%%% GOL CA initialized as Pulsar
soln=GOL.solveGOL(system=9, duration=30)
GOL.animateGOL(soln, out=f'{gifpath}/pulsar.gif')
pulsar = GOL.plotDensity(soln)
pulsar[1].set_title('Density with initial pulsar')

#%%% GOL CA initialized as Glider
soln=GOL.solveGOL(system=11, duration=30)
GOL.animateGOL(soln, out=f'{gifpath}/glider.gif')
glider = GOL.plotDensity(soln)
glider[1].set_title('Density with initial glider')

#%%% GOL CA initialized as Die Hard
soln=GOL.solveGOL(system=16, duration=30)
GOL.animateGOL(soln, out=f'{gifpath}/diehard.gif')
diehard = GOL.plotDensity(soln)
diehard[1].set_title('Density with initial diehard')