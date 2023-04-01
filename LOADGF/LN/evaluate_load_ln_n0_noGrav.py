# *********************************************************************
# FUNCTION TO COMPUTE LOAD LOVE NUMBERS FROM Y-SOLUTIONS (n=0)
# SPECIAL CASE OF NO GRAVITY
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

def main(n,Y_load,a,gs,T_sc,L_sc):

    # Extract Solutions
    Y1sol_load = Y_load[0]
    Y2sol_load = Y_load[1]

    # Compute Load Love Numbers
    hprime = Y1sol_load
    nlprime = np.array([[0.]])

    # Flatten Arrays
    hprime  = hprime.flatten()
    nlprime = nlprime.flatten()
    nkprime = np.multiply(nlprime,0.)

    # Return Variables
    return hprime,nlprime,nkprime


