# *********************************************************************
# FUNCTION TO COMPUTE THE PARTIAL DERIVATIVES OF LOVE NUMBERS
# See: Okubo & Saito (1983); Martens et al. (2016, JGR-Solid Earth)
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

# Import Python Modules
from __future__ import print_function
import numpy as np
from scipy.integrate import simps
import math
import sys
from LOADGF.PL import dLagrangian_solid

def main(n,sint,Y1,Y2,YP1,YP2,piG,a_norm,rho_norm,g_norm):

    # Define the Normalization Factor for I (See Eq. 17 & Table 2 in Okubo & Saito 1983)
    nf = -(4.*piG)/((2.*n+1.)*a_norm)

    # :: Compute dL/dK, dL/dmu, and dL/drho (See Table 2 in Okubo & Saito 1983) 

    # Separate Out Components
    x1dot = YP1[:,0]
    x3dot = YP1[:,2]
    x5dot = YP1[:,4]
    y1dot = YP2[:,0]
    y3dot = YP2[:,2]
    y5dot = YP2[:,4]
    x1    = Y1[:,0]
    x3    = Y1[:,2]
    x5    = Y1[:,4]
    y1    = Y2[:,0]
    y3    = Y2[:,2]
    y5    = Y2[:,4]
    X = 2.*x1 - (n*(n+1))*x3
    Y = 2.*y1 - (n*(n+1))*y3

    # Compute the Lagrangian Partial Derivatives
    dL_dK,dL_dmu,dL_drho = dLagrangian_solid.main(n,sint,x1,x3,x5,y1,y3,y5,x1dot,x3dot,x5dot,y1dot,y3dot,y5dot,piG,rho_norm,g_norm)
 
    # For Each Value of 'sint', Integrate Extra Argument to the Surface (See O&S Eq. 17 for pi=rho)
    integrand = np.multiply(np.divide(rho_norm,sint), (np.multiply(x1,Y) + np.multiply(y1,X)))
    rho_integral = dL_drho.copy()
    for ii in range(0,len(sint)-1):
        myXrng = sint[ii:-1]
        myYrng = integrand[ii:-1]
        rho_integral[ii] = simps(myYrng,myXrng)
    rho_integral[-1] = 0.

    # :: Compute dI with Respect to dK, dmu, and drho
    dI_dK  = nf*dL_dK
    dI_dmu = nf*dL_dmu
    dI_drho = nf*(dL_drho - np.multiply(rho_integral,np.power(sint,2.))*4.*piG)

    # Return Derivatives of I (Okubo & Saito Eq. 17)
    return dI_dK, dI_dmu, dI_drho


