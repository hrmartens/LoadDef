# *********************************************************************
# FUNCTION TO COMPUTE LEGENDRE POLYNOMIALS BY RECURSION RELATIONS
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
import sys

def main(n,myt):

    # Compute Argument and Derivative
    myx  = np.cos(myt)
    mydx = -np.sin(myt)
    # Initialize Arrays
    P   = np.zeros(len(n))
    dP  = np.zeros(len(n))
    ddP = np.zeros(len(n)) 

    # Initialize Counter
    count = 0

    # Ensure First Value is n=0
    if (n[0] != 0):
        sys.exit('Error: Cannot Compute Legendre Recursion Since an n=0 Term Was Not Included (Greens Functions Would Be Inaccurate).')

    # Loop Through All n
    for jj in range(0,len(n)):

        # Current n
        myn = n[jj]

        # Determine P and dP from Recurrence Relations
        if (myn == 0):
            P[count]  = 1.
            dP[count] = 0.
        elif (myn == 1):
            P[count]  = myx
            dP[count] = mydx
        else:
            P[count]  = ((2.*myn-1.)/myn)*myx*P[(count-1)] - \
                ((myn-1.)/myn)*P[(count-2)]
            dP[count] = myn*(P[count-1] - myx*P[count])/mydx 

        # Compute ddP from Legendre's Differential Equation
        ddP[count] = (myx/mydx)*dP[count] - (myn*(myn+1))*P[count]

        # Update Counter
        count = count + 1

    # Return Legendre Polynomials and Derivatives
    return P,dP,ddP

