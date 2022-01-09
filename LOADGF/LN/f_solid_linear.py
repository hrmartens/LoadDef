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
import numba


def main(si,Y,n,model_radii,model_lmrg,wnd,ond,piG,m):
    # ODE seems not to like calling a numba compiled function directly; hence, some boilerplate
    return dYdr(si,Y,n,model_radii,model_lmrg,wnd,ond,piG,m)

# just in time compile this function to C
@numba.jit(
    numba.float64[:](
        numba.float64,      # define all types to help the compiler optimize
        numba.float64[:],
        numba.float64,
        numba.float64[:],
        numba.float64[:,:],
        numba.float64,
        numba.float64,
        numba.float64,
        numba.int64,
    ),
    cache=True,     # avoid recompilation after resarting python
    nopython=True,  # avoid callbacks to python
)
def dYdr(si,Y,n,model_radii,model_lmrg,wnd,ond,piG,m):

    # create workspace to avoid allocations on the c-side
    YP = np.zeros(18)

    # find layer in the model
    idx = np.searchsorted(model_radii, si)

    # hand written linear interpolation
    r1 = model_radii[idx-1]
    r2 = model_radii[idx]
    # test (gravity varies through to the surface; gravity is in fourth column of model_lmrg)
    # Y3 = m * X3 + b
    #    = (Y2 - Y1)/(X2 - X1) * X3 + Y2 - (Y2 - Y1)/(X2 - X1) * X2
    #    = (Y2-Y1)*X3/(X2-X1) + (X2-X1)*Y2/(X2-X1) - (Y2-Y1)*X2/(X2-X1)
    #    = (Y2X3 - Y1X3 + X2Y2 - X1Y2 - X2Y2 + X2Y1) / (X2-X1)
    #    = (Y2X3 - Y2X1 - Y1X3 + Y1X2) / (X2-X1)
    #    = (Y1 * (X2-X3) + Y2 * (X3-X1)) / (X2-X1)
    #    = equation written below
    #gnd1 = model_lmrg[idx-1][3]
    #gnd2 = model_lmrg[idx][3]
    #slope = (gnd2-gnd1)/(r2-r1)
    #intercept = gnd2 - slope*r2
    #gnd_of_interest = slope*si + intercept
    # interpolate parameters
    lndi, mndi, rndi, gndi = (
        (model_lmrg[idx-1] * (r2 - si) + model_lmrg[idx] * (si - r1)) /
        (r2 - r1)
    )

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
    # loop over the three independent solutions
    # use simple loops instead of numpy functions to avoid callbacks
    # these loops compile very well in the jit
    for i in range(3):
        offset = 6 * i
        YP[0+offset] = c11 * Y[0+offset] + c12 * Y[1+offset] + c13 * Y[2+offset]
        YP[1+offset] = c21 * Y[0+offset] + c22 * Y[1+offset] + c23 * Y[2+offset] + c24 * Y[3+offset] + c26 * Y[5+offset]
        YP[2+offset] = c31 * Y[0+offset] + c33 * Y[2+offset] + c34 * Y[3+offset]
        YP[3+offset] = c41 * Y[0+offset] + c42 * Y[1+offset] + c43 * Y[2+offset] + c44 * Y[3+offset] + c45 * Y[4+offset]
        YP[4+offset] = c51 * Y[0+offset] + c56 * Y[5+offset]
        YP[5+offset] = c63 * Y[2+offset] + c65 * Y[4+offset] + c66 * Y[5+offset]

    return YP

