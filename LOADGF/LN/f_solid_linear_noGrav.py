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
    YP = np.zeros(8)

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

    # USE PROPAGATOR MATRIX TECHNIQUE TO COMPUTE dY/dr
    # loop over the three independent solutions
    # use simple loops instead of numpy functions to avoid callbacks
    # these loops compile very well in the jit
    for i in range(2):
        offset = 4 * i
        YP[0+offset] = c11 * Y[0+offset] + c12 * Y[1+offset] + c13 * Y[2+offset]
        YP[1+offset] = c21 * Y[0+offset] + c22 * Y[1+offset] + c23 * Y[2+offset] + c24 * Y[3+offset] 
        YP[2+offset] = c31 * Y[0+offset] + c33 * Y[2+offset] + c34 * Y[3+offset]
        YP[3+offset] = c41 * Y[0+offset] + c42 * Y[1+offset] + c43 * Y[2+offset] + c44 * Y[3+offset] 

    return YP

