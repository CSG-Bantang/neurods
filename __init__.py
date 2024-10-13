#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:53:25 2024

@author: reinierramos
"""

from .hhodes import (odes, Iext)
from .bte import (alphah, alpham, alphan, betah, betam, betan)
from .sschannels import (m_inf, h_inf, n_inf, tau_m, tau_h, tau_n)
from .hhsolvers import (lsoda, euler, rk4)
from .hhanalysis import (solveHH, asymptoticChannels, timeConstants, firingRate)
from .hhplotter import *