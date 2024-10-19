#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 11:48:48 2024

@author: reinierramos
"""

import numpy as np

def updateLattice(rule:int, L:int, lattice:list[int], mid:int) -> list[int]:
    """
    Updates the ECA lattice applying ecaRules.

    """
    prev = lattice.copy()
    for i in range(L):
        rolled = np.roll(prev, mid-i)
        neighbors = rolled[mid-1:mid+2]
        nbc = ''.join([str(nei) for nei in neighbors])
        lattice[i] = ecaRules(rule, nbc)
    return lattice

def ecaRules(rule:int, nbc:str) -> int:
    """
    Rule is converted into 8-digit binary, where each digit is mapped into one 
    of the possible states in nbc.
    
    Example
    -------
    States = set(0, 1)
    Possible nbc given the States = set(111, 110, 101, 100, 011, 010, 001, 000) 
    Rule = 30 ->bin-> '01111000'  = set( 0,   1,   1,   1,   1,   0,   0,   0 )
    Mapping is given by the same index in both sets.
    Only the middle digit in given in nbc is changed into the digit from rule.
    
    """
    BinRule = f'{bin(rule)[2:]:0>8}'[::-1]
    nbcList = [bin(i)[2:].zfill(3) for i in range(8)]
    ruleDict = dict(zip(nbcList, BinRule))
    return int(ruleDict.get(nbc))