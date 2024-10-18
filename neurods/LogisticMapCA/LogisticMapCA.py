#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:55:59 2024

@author: reinierramos
"""

from .lmcaanalysis import (solveLM, logisticEquation, 
                           logisticReturnMap, solveLMCA)

from .lmcaplotter import (plotXvsT, plotReturnMap, plotDensity,
                          animateLMCA, animate3DLMCA)