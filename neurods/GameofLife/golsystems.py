#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:29:06 2024

@author: reinierramos
"""

import numpy as np

#### Predefined GOL Patterns

def initializeGOL(system:int):
    """
    Returns initial GOL pattern according to system.

    Parameters
    ----------
    system : int
    
    Allowed values are
    1 : block, still-life
    2 : beehive, still-life
    3 : loaf, still-life
    4 : boat, still-life
    5 : tub, still-life
    6 : blinker, period 2
    7 : toad, period 2
    8 : beacon, period 2
    9 : pulsar, period 3
    10 : pentadecathlon, period 15
    11 : glider
    12 : lightweight spaceship, LWSS
    13 : middleweight spaceship, MWSS
    14 : heavyweight spaceship, HWSS
    15 : R-pentomino, Methuselah
    16 : Die hard, Methuselah
    17 : Acorn, Methuselah

    """
    GOLSystems = {
    1:  np.array([0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0], dtype=np.int32), ## block
    2:  np.array([0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,1,0,0,0,
                  1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.int32), ## beehive
    3:  np.array([0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,1,0,0,0,
                  1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0], dtype=np.int32), ## loaf
    4:  np.array([0,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,1,0,0,
                  0,0,0,0,0],                       dtype=np.int32), ## boat
    5:  np.array([0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,
                  0,0,0,0,0],                       dtype=np.int32), ## tub
    6:  np.array([0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,
                  0,0,0,0,0],                       dtype=np.int32), ## blinker-2
    7:  np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,
                  1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.int32), ## toad-2
    8:  np.array([0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,
                  0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0], dtype=np.int32), ## beacon-2
    9:  np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
                  1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,
                  0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,
                  0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,
                  0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,
                  0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,1,0,1,
                  0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,
                  0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0],               dtype=np.int32), ## pulsar-3
    10: np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,
                  0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,0,1,1,0,0,
                  0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0],                         dtype=np.int32), ## pentadecathlon-15
    11: np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,1,0,
                  0,0,0,0,0],                       dtype=np.int32), ## glider
    12: np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
                  0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,
                  0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0],                         dtype=np.int32), ## lightweight spaceship
    13: np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,
                  0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,
                  0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0],                         dtype=np.int32), ## middleweight spaceship
    14: np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
                  0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,
                  0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,
                  1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0],                               dtype=np.int32), ## heavyweight spaceship
    15: np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,
                  1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0],                               dtype=np.int32), ## R-pentomino
    16: np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
                  0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,
                  1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0],                         dtype=np.int32), ## Die hard
    17: np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                  1,0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0],                               dtype=np.int32)  ## Acorn
    }
    return GOLSystems.get(system)