# *********************************************************************
# FUNCTION TO COMPUTE POTENTIAL LOVE NUMBERS FROM Y-SOLUTIONS (n>=1)
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

def main(n,Y_pot,a,gs,T_sc,L_sc):

    # Extract Solutions
    Y1sol_pot = Y_pot[0]
    Y2sol_pot = Y_pot[1]
    Y3sol_pot = Y_pot[2]
    Y4sol_pot = Y_pot[3]
    Y5sol_pot = Y_pot[4]
    Y6sol_pot = Y_pot[5]

    # Compute Potential Love Numbers
    hpot = Y1sol_pot 
    nlpot = n*Y3sol_pot
    Y5sol = Y5sol_pot*((L_sc**2.)*(T_sc**(-2.)))
    nkpot = n*((Y5sol/(a*gs))-1.)

    # Force all potential Love numbers of degree n=1 to NaN
    # Undefined in static case [Okubo & Endo 1986, Saito 1974]
    # Will be considered futher in a future release of LoadDef
    if (n == 1):
        hpot = np.asarray(np.nan)
        nlpot = np.asarray(np.nan)
        nkpot = np.asarray(np.nan)
        #hpot = hpot - nkpot
        #nlpot = nlpot - nkpot
        #nkpot = nkpot - nkpot

    # Flatten Arrays
    hpot    = hpot.flatten()
    nlpot   = nlpot.flatten()
    nkpot   = nkpot.flatten() 

    # Return Variables
    return hpot,nlpot,nkpot


