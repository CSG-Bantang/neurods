#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:34:49 2024

@author: reinierramos
"""

from neurods import BriansBrain as BB
import os

fireRules = {'=':'=', '<=':r'\leq', '>=':r'\geq', '<':'<', '>':'>'}

#%% Brian's Brain Systems

gifpath = './BB-gifs'
os.makedirs(gifpath, exist_ok=True)

#%%% Original BB CA
soln=BB.solveBB()
BB.animateBB(soln, out=f'{gifpath}/original.gif')
mooreouter = BB.plotDensity(soln)
mooreouter[1].set_title("Original Brian's Brain")

#%%% BB CA with modified neigborhood and initial state
dq = 0.5
df = 0.3
k  = 2
soln=BB.solveBB(system=2, df=df, dq=dq, k=k)
BB.animateBB(soln, out=f'{gifpath}/modified-nbc-and-init.gif')
modnbcinit = BB.plotDensity(soln)
modnbcinit[1].set_title(f"BB, dq={dq}, df={df}, k={k}")

#%%% BB CA with modified firing rule
Lambda = 3
firingRule = '<='
soln=BB.solveBB(Lambda=Lambda, firingRule=firingRule)
BB.animateBB(soln, out=f'{gifpath}/modified-firing-rule.gif')
modfiring = BB.plotDensity(soln)
modfiring[1].set_title(fr"$\Lambda {fireRules.get(firingRule)} {Lambda}$")

#%%% BB CA with modified refractory rule
tRefrac = 2
soln=BB.solveBB(tRefrac=tRefrac)
BB.animateBB(soln, out=f'{gifpath}/modified-refractory-rule.gif')
modrefrac = BB.plotDensity(soln)
modrefrac[1].set_title(fr"$ t_{{\rm refrac}} = {tRefrac}$")