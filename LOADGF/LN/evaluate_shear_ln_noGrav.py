# *********************************************************************
# FUNCTION TO COMPUTE SHEAR LOVE NUMBERS FROM Y-SOLUTIONS (n>=1)
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

def main(n,Y_shr,a,gs,T_sc,L_sc):

    # Extract Solutions
    Y1sol_shr = Y_shr[0]
    Y2sol_shr = Y_shr[1]
    Y3sol_shr = Y_shr[2]
    Y4sol_shr = Y_shr[3]

    # Compute Shear Love Numbers
    hshr = Y1sol_shr
    nlshr = n*Y3sol_shr

    # Force all shear Love numbers of degree n=1 to NaN
    # Undefined in static case [Okubo & Endo 1986, Saito 1974]
    # Will be considered futher in a future release of LoadDef
    if (n == 1):
        hshr = np.asarray(np.nan)
        nlshr = np.asarray(np.nan)
        nkshr = np.asarray(np.nan)

    # Flatten Arrays
    hshr    = hshr.flatten()
    nlshr   = nlshr.flatten()
    nkshr   = np.multiply(nlshr,0.)

    # Return Variables
    return hshr,nlshr,nkshr


