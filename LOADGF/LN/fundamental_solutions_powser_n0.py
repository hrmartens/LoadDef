# *********************************************************************
# FUNCTION TO COMPUTE STARTING SOLUTIONS USING A POWER-SERIES EXPANSION
# OF THE EQUATIONS OF MOTION FOR SPHEROIDAL OSCILLATION (n=0) 
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

# FIND THE INITIAL SOLUTIONS BY POWER SERIES EXPANSION
# SEE, E.G., SMYLIE (2013) OR CROSSLEY (1975)

import numpy as np
from scipy import interpolate

def main(si,tck_lnd,tck_mnd,tck_rnd,wnd,ond,piG):

    # Interpolate Parameters at Current Radius
    lndi = interpolate.splev(si,tck_lnd,der=0)
    rndi = interpolate.splev(si,tck_rnd,der=0)
    mndi = interpolate.splev(si,tck_mnd,der=0)

    # Compute Non-Dimensional Gamma
    gamnd = (4./3)*piG*rndi - (2./3)*(ond**2)

    # Initial Values for First Fundamental Solution
    z1_r0 = 1
    z2_r0 = (3.*lndi+2.*mndi)
    z5_r0 = 2.*piG*rndi
    z6_r0 = 0
    Y1a = np.array([z1_r0,z2_r0,z5_r0,z6_r0]) 
  
    # Coefficients of Second Terms in Expansion of
    # First Fundamental Solution
    z1_r2 = -(rndi*(4.*gamnd+(wnd**2)+(2.*(ond**2))))/(10.*(lndi+2.*mndi))
    z2_r2 = (5.*lndi+6.*mndi)*z1_r2
    z5_r2 = piG*rndi*z1_r2
    z6_r2 = 0
    Y1b = np.array([z1_r2,z2_r2,z5_r2,z6_r2]) 

    # Coefficients of Third Terms in Expansion of
    # First Fundamental Solution
    z1_r4 = -((rndi*(4.*gamnd+(wnd**2)+(2.*(ond**2))))/(28.*(lndi+2.*mndi)))*z1_r2
    z2_r4 = (7.*lndi+10.*mndi)*z1_r4
    z5_r4 = (2./3)*piG*rndi*z1_r4
    z6_r4 = 0
    Y1c = np.array([z1_r4,z2_r4,z5_r4,z6_r4]) 

    # Compute the Expansion for the First Fundamental Solution (Free Constant A11)
    # Smylie (2013) PG.S 244-245
    Z1nd = Y1a + Y1b*(si**2) + Y1c*(si**4)

    # Fill Solution Matrix
    Z = np.zeros((2,4))
    Z[0,2] = 1
    Z[1,:] = Z1nd

    # Convert to Y Solutions
    Y = Z.copy()
    alpha = 0.
    Y[0,0] = Z[0,0]*(si**(alpha+1.)) # Y1
    Y[0,1] = Z[0,1]*(si**alpha)     # Y2
    Y[0,2] = Z[0,2]*(si**(alpha+2.)) # Y5
    Y[0,3] = Z[0,3]*(si**(alpha+1.)) # Y6
    Y[1,0] = Z[1,0]*(si**(alpha+1.)) # Y1
    Y[1,1] = Z[1,1]*(si**alpha)     # Y2
    Y[1,2] = Z[1,2]*(si**(alpha+2.)) # Y5
    Y[1,3] = Z[1,3]*(si**(alpha+1.)) # Y6

    # Return Solutions
    return Y

