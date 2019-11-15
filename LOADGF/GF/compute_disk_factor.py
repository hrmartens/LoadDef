# *********************************************************************
# FUNCTION TO COMPUTE A DISK FACTOR FOR LOAD GREEN'S FUNCTIONS
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

from __future__ import print_function
import numpy as np
from LOADGF.GF import compute_legendre

def main(n,alpha):

    # Compute Legendre Polynomials for Theta=Alpha
    P,dP,ddP = compute_legendre.main(n,alpha)

    # Initialize Array and Counter
    dfac = np.zeros(len(n))
    count = 0

    # Loop Through All n
    for jj in range(0,len(n)):

        # Current n
        myn = n[jj]

        # Compute Disk Factor
        if (myn == 0):
            dfac[count] = 1.
        else:
            mydP = dP[jj]
            dfac[count] = ( -(1. + np.cos(alpha)) / (myn*(myn+1.) * np.sin(alpha)) ) * mydP	

        # Update Counter
        count = count + 1
  
    # Return Disk Factor
    return dfac

