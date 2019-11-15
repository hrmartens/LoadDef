# *********************************************************************
# FUNCTION TO COMPUTE ASYMPTOTIC LOAD LOVE NUMBERS
#
# Copyright (c) 2014-2019: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
#
# This file is part of LoadDef.
#
#    LoadDef is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    LoadDef is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with LoadDef.  If not, see <https://www.gnu.org/licenses/>.
#
# *********************************************************************

import numpy as np
import math
import sys

def main(myn,piG,lnd,mnd,gnd,rnd,a,L_sc):
    
    # Compute the Asymptotic Load Love Numbers 
    # Farrell, 1972
    if isinstance(lnd,np.ndarray): # array of values
        ls = lnd[-1]
        ms = mnd[-1]
        gsnd = gnd[-1]
        rs = rnd[-1]
    else: 
        ls = lnd; ms = mnd; gsnd = gnd; rs = rnd
    Rs = a/L_sc
    h_inf = -((gsnd**2)*(ls+2.*ms))/(4.*piG*ms*(ls+ms))
    l_inf = (gsnd**2)/(4.*piG*(ls+ms))
    k_inf = (-Rs*rs*gsnd)/(2.*ms)

    # Compute the More Accurate Asymptotic Expressions, N->INF
    # Guo et al. (2004)
    h_inf_prime = ((gsnd**2)/(4.*piG*(ls+ms)))*(-(ms/(ls+ms))+ \
        (Rs*rs*gsnd*(ls**2 + ls*ms - ms**2)/(2.*(ms**2)*(ls+ms)))+ \
        (2.*piG*Rs*rs*(ls+ms)/(gsnd*ms)))
    l_inf_prime = ((gsnd**2)/(4.*piG*(ls+ms)))* \
        ((-(3.*(ls**2)+8.*ls*ms+3.*(ms**2))/(2.*ms*(ls+ms)))+ \
        (Rs*rs*gsnd*(ls+2.*ms)/(2.*ms*(ls+ms)))) 
    k_inf_prime = (Rs*gsnd*rs/ms)*((ls/(4.*(ls+ms)))+ \
        (Rs*rs*gsnd*(2.*ls+ms)/(8.*ms*(ls+ms))) + (piG*Rs*rs/gsnd))

    # Combine the Asymptotic Expressions (Guo et al. 2004)
    if (myn[0] == 0):
        myn[0] = np.NAN
    hprime_asym = h_inf + (1./myn)*h_inf_prime
    nlprime_asym = l_inf + (1./myn)*l_inf_prime
    nkprime_asym = k_inf + (1./myn)*k_inf_prime
    if (math.isnan(myn[0]) == True):
        myn[0] = 0

    # Return Values
    return hprime_asym,nkprime_asym,nlprime_asym,h_inf, \
	h_inf_prime,l_inf,l_inf_prime,k_inf,k_inf_prime

