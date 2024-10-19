#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:30:07 2024

@author: reinierramos
"""

from neurods import WolframECA as WECA

rule = 30
soln = WECA.solveECA(rule=rule, init=1)
evolution = WECA.plotECA(soln)
evolution[1].set_title(f'Rule {rule} initialized at the middle')

middle = WECA.plotDensity(soln)
middle[1].set_title(f'Rule {rule} initialized at the middle')

rule = 90
soln = WECA.solveECA(rule=rule, init=0, dens1=0.05)
evolution = WECA.plotECA(soln)
evolution[1].set_title(f'Rule {rule} initialized at random')

middle = WECA.plotDensity(soln)
middle[1].set_title(f'Rule {rule} initialized at random')
