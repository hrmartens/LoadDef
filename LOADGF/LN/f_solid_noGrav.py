# *********************************************************************
# FUNCTION TO PROPAGATE Y-SOLUTIONS THROUGH A SOLID LAYER (n>=1)
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
from scipy import interpolate

def main(si,Y,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m):

    # Interpolate Parameters at Current Radius
    lndi = interpolate.splev(si,tck_lnd,der=0)
    rndi = interpolate.splev(si,tck_rnd,der=0)
    mndi = interpolate.splev(si,tck_mnd,der=0)
    gndi = interpolate.splev(si,tck_gnd,der=0)

    # Compute n1, Beta, Delta, and Epsilon Parameters
    # See Smylie (2013)
    n1 = n*(n+1.)
    bndi = 1./(lndi+2.*mndi)
    dndi = 2.*mndi*(3.*lndi + 2.*mndi)*bndi
    endi = 4.*n1*mndi*(lndi+mndi)*bndi - 2.*mndi

    # Build A Matrix (where dY/dr = A Y)
    # See Smylie (2013)
    # :: Eqs. 90/91 in Takeuchi & Saito (1972)
    # :: See also Eq. 30 in T&S
    # :: Compared with Eq. 82 in T&S
    # :: Removed y5 and y6 (rows & columns)
    # :: Removed all terms involving g
    # :: Comparing T&S Eq. 91 with Alterman (1959),
    # :: y5 and y6 disappear, along with all terms
    # :: involving g (but not rho on its own)
    # :: Dahlen & Tromp (1998), Sec. 8.8.6, pg. 295:
    # :: For the case of neglecting self-gravity,
    # :: set G = g = y5 = y6 = 0.
    c11 = -2.*lndi*bndi/si
    c12 = bndi
    c13 = n1*lndi*bndi/si
    c14 = 0.

    c21 = (2.*dndi/(si**2)) - (rndi*(wnd**2))
    c22 = -4.*mndi*bndi/si
    c23 = n1*(-dndi/(si**2))
    c24 = n1/si

    c31 = -1./si
    c32 = 0.
    c33 = 1./si
    c34 = 1./mndi

    c41 = -dndi/(si**2)
    c42 = -lndi*bndi/si
    c43 = endi/(si**2) - rndi*(wnd**2)
    c44 = -3./si

    # Matrix of Parameters
    A = np.array([[c11, c12, c13, c14], \
        [c21, c22, c23, c24], \
        [c31, c32, c33, c34], \
        [c41, c42, c43, c44]])
    
    # USE PROPAGATOR MATRIX TECHNIQUE TO COMPUTE dY/dr
    # Only one Y solution computed at a time (still the case for computing partial derivatives)
    if (len(Y) == 4): 
        YP = np.dot(A,Y)
    # Updated code (both solutions together)
    else:
        YP1 = np.dot(A,Y[0:4])
        YP2 = np.dot(A,Y[4:8])
        YP = np.concatenate([YP1,YP2])

    # RETURN UPDATED DERIVATIVES
    return YP

