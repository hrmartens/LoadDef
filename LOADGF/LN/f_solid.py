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
import numba
import time


#def main(si,Y,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m):
def main(si,Y,n,tck_lmrg,wnd,ond,piG,m):

    # slow part is now this evaluation
    # Interpolate Parameters at Current Radius
    #lndi = float(interpolate.splev(si,tck_lnd,der=0))
    #rndi = float(interpolate.splev(si,tck_rnd,der=0))
    #mndi = float(interpolate.splev(si,tck_mnd,der=0))
    #gndi = float(interpolate.splev(si,tck_gnd,der=0))

    # a = time.time()

    lndi, rndi, mndi, gndi = map(float, interpolate.splev(si, tck_lmrg))

    # b = time.time()

    YP = np.zeros(18)
    #YP = np.zeros(6)
    dYdr(si,Y,n,wnd,ond,piG,m,lndi, rndi, mndi, gndi, YP)

    # c = time.time()
    # print((b-a)/(c-b))

    return YP


@numba.jit(
    numba.void(
        numba.float64,
        numba.float64[:],
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.int64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64[:],
    ),
    cache=True,
    nopython=True,
)
def dYdr(si,Y,n,wnd,ond,piG,m, lndi, rndi, mndi, gndi, YP):

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

    c21 = (-2.*rndi/si)*(2.*gndi+si*(ond**2))+ \
        (2.*dndi/(si**2)) - (rndi*(wnd**2))
    c22 = -4.*mndi*bndi/si
    c23 = n1*(rndi*gndi/si - dndi/(si**2)) + (2.*m*wnd*ond*rndi)
    c24 = n1/si
    c26 = -rndi

    c31 = -1./si
    c32 = 0.
    c33 = 1./si
    c34 = 1./mndi

    c41 = rndi*gndi/si - dndi/(si**2) + 2.*m*wnd*ond*rndi/n1
    c42 = -lndi*bndi/si
    c43 = endi/(si**2) - rndi*(wnd**2) + 2*m*wnd*ond*rndi/n1
    c44 = -3./si
    c45 = -rndi/si

    c51 = 4.*piG*rndi
    c56 = 1.

    c63 = -4.*piG*rndi*n1/si
    c65 = n1/(si**2)
    c66 = -2./si

    # USE PROPAGATOR MATRIX TECHNIQUE TO COMPUTE dY/dr
    YP[0] = c11 * Y[0] + c12 * Y[1] + c13 * Y[2]
    YP[1] = c21 * Y[0] + c22 * Y[1] + c23 * Y[2] + c24 * Y[3] + c26 * Y[5]
    YP[2] = c31 * Y[0] + c33 * Y[2] + c34 * Y[3]
    YP[3] = c41 * Y[0] + c42 * Y[1] + c43 * Y[2] + c44 * Y[3] + c45 * Y[4]
    YP[4] = c51 * Y[0] + c56 * Y[5]
    YP[5] = c63 * Y[2] + c65 * Y[4] + c66 * Y[5]

    YP[0+6] = c11 * Y[0+6] + c12 * Y[1+6] + c13 * Y[2+6]
    YP[1+6] = c21 * Y[0+6] + c22 * Y[1+6] + c23 * Y[2+6] + c24 * Y[3+6] + c26 * Y[5+6]
    YP[2+6] = c31 * Y[0+6] + c33 * Y[2+6] + c34 * Y[3+6]
    YP[3+6] = c41 * Y[0+6] + c42 * Y[1+6] + c43 * Y[2+6] + c44 * Y[3+6] + c45 * Y[4+6]
    YP[4+6] = c51 * Y[0+6] + c56 * Y[5+6]
    YP[5+6] = c63 * Y[2+6] + c65 * Y[4+6] + c66 * Y[5+6]

    YP[0+12] = c11 * Y[0+12] + c12 * Y[1+12] + c13 * Y[2+12]
    YP[1+12] = c21 * Y[0+12] + c22 * Y[1+12] + c23 * Y[2+12] + c24 * Y[3+12] + c26 * Y[5+12]
    YP[2+12] = c31 * Y[0+12] + c33 * Y[2+12] + c34 * Y[3+12]
    YP[3+12] = c41 * Y[0+12] + c42 * Y[1+12] + c43 * Y[2+12] + c44 * Y[3+12] + c45 * Y[4+12]
    YP[4+12] = c51 * Y[0+12] + c56 * Y[5+12]
    YP[5+12] = c63 * Y[2+12] + c65 * Y[4+12] + c66 * Y[5+12]
