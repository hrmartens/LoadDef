# *********************************************************************
# FUNCTION TO PROPAGATE Y-SOLUTIONS THROUGH A SOLID LAYER (n>=1)
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
    c11 = -2.*lndi*bndi/si
    c12 = bndi
    c13 = n1*lndi*bndi/si
    c14 = 0.
    c15 = 0.
    c16 = 0.

    c21 = (-2.*rndi/si)*(2.*gndi+si*(ond**2))+ \
        (2.*dndi/(si**2)) - (rndi*(wnd**2))
    c22 = -4.*mndi*bndi/si
    c23 = n1*(rndi*gndi/si - dndi/(si**2)) + (2.*m*wnd*ond*rndi)
    c24 = n1/si
    c25 = 0.
    c26 = -rndi

    c31 = -1./si
    c32 = 0.
    c33 = 1./si
    c34 = 1./mndi
    c35 = 0.
    c36 = 0.

    c41 = rndi*gndi/si - dndi/(si**2) + 2.*m*wnd*ond*rndi/n1
    c42 = -lndi*bndi/si
    c43 = endi/(si**2) - rndi*(wnd**2) + 2*m*wnd*ond*rndi/n1
    c44 = -3./si
    c45 = -rndi/si
    c46 = 0. 

    c51 = 4.*piG*rndi
    c52 = 0.
    c53 = 0.
    c54 = 0.
    c55 = 0.
    c56 = 1.

    c61 = 0.
    c62 = 0.
    c63 = -4.*piG*rndi*n1/si
    c64 = 0.
    c65 = n1/(si**2)
    c66 = -2./si

    # Matrix of Parameters
    A = np.array([[c11, c12, c13, c14, c15, c16], \
        [c21, c22, c23, c24, c25, c26], \
        [c31, c32, c33, c34, c35, c36], \
        [c41, c42, c43, c44, c45, c46], \
        [c51, c52, c53, c54, c55, c56], \
        [c61, c62, c63, c64, c65, c66]])

    # USE PROPAGATOR MATRIX TECHNIQUE TO COMPUTE dY/dr
    # Original code
    #YP = np.dot(A,Y)
    if (len(Y) == 6): 
        YP = np.dot(A,Y)
    # Updated code (all three solutions together)
    else:
        YP1 = np.dot(A,Y[0:6])
        YP2 = np.dot(A,Y[6:12])
        YP3 = np.dot(A,Y[12:18])
        YP = np.concatenate([YP1,YP2,YP3])

    # RETURN UPDATED DERIVATIVES
    return YP

