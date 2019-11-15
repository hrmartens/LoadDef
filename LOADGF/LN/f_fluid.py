# *********************************************************************
# FUNCTION TO PROPAGATE Y-SOLUTIONS THROUGH FLUID LAYER (n>=1)
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

def main(si,Y,n,tck_lnd,tck_rnd,tck_gnd,wnd,piG,m,ond):

    # Interpolate Parameters at Current Radius
    lndi = interpolate.splev(si,tck_lnd,der=0)
    rndi = interpolate.splev(si,tck_rnd,der=0)
    gndi = interpolate.splev(si,tck_gnd,der=0)

    # Compute n1
    n1 = n*(n+1.)

    # Solve for Y3
    # See Smylie (2013), PG. 230, EQ. 3.116
    # Coefficients of Y1 in equation for Y3
    Y3_Y1 = (1./((wnd**2)*rndi - (2.*m*wnd*ond*rndi/n1))) * \
	(rndi*gndi/si + 2.*m*wnd*ond*rndi/n1)
    # Coefficients of Y2 in equation for Y3
    Y3_Y2 = (1./((wnd**2)*rndi - (2.*m*wnd*ond*rndi/n1))) * \
	(-1./si)
    # Coefficients of Y5 in equation for Y3
    Y3_Y5 = (1./((wnd**2)*rndi - (2.*m*wnd*ond*rndi/n1))) * \
    	(-rndi/si)

    # Build A Matrix (where dY/dr = A Y)
    c11 = (-2./si) + (n1/si)*Y3_Y1
    c12 = (1./lndi) + (n1/si)*Y3_Y2
    c15 = (n1/si)*Y3_Y5
    c16 = 0.

    c21 = (-2.*rndi/si)*(2.*gndi + si*(ond**2)) + (n1*rndi*gndi/si)*Y3_Y1 - \
	(wnd**2)*rndi + (2.*m*wnd*ond*rndi)*Y3_Y1
    c22 = (n1*rndi*gndi/si)*Y3_Y2 + (2.*m*wnd*ond*rndi)*Y3_Y2
    c25 = (n1*rndi*gndi/si)*Y3_Y5 + (2.*m*wnd*ond*rndi)*Y3_Y5
    c26 = -rndi

    c51 = 4.*piG*rndi
    c52 = 0.
    c55 = 0.
    c56 = 1.

    c61 = (-4.*piG*rndi*n1/si)*Y3_Y1
    c62 = (-4.*piG*rndi*n1/si)*Y3_Y2
    c65 = (-4.*piG*rndi*n1/si)*Y3_Y5 + (n1/(si**2))
    c66 = -2./si

    # Matrix of Parameters
    A = np.array([[c11, c12, c15, c16], \
        [c21, c22, c25, c26], \
        [c51, c52, c55, c56], \
        [c61, c62, c65, c66]])

    # Compute dY/dr
    YP = np.dot(A,Y)

    # Return Updated Derivatives
    return YP

