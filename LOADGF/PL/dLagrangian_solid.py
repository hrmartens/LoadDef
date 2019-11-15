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

import numpy as np

def main(n,sint,x1,x3,x5,y1,y3,y5,x1dot,x3dot,x5dot,y1dot,y3dot,y5dot,piG,rho_norm,g_norm):

    X = 2.*x1 - (n*(n+1))*x3
    Y = 2.*y1 - (n*(n+1))*y3
    dL_dK = np.multiply(np.multiply(np.power(sint,2.),x1dot),y1dot) + \
        np.multiply(np.multiply(x1dot,Y),sint) + np.multiply(np.multiply(y1dot,X),sint) + np.multiply(X,Y)
    dL_dmu = (4./3.)*np.multiply(np.multiply(np.power(sint,2.),x1dot),y1dot) - \
        (2./3.)*(np.multiply(np.multiply(x1dot,Y),sint) + np.multiply(np.multiply(y1dot,X),sint)) + \
        (1./3.)*np.multiply(X,Y) + (n*(n+1))*(np.multiply((np.multiply(sint,y3dot) + y1 - y3),\
        (np.multiply(sint,x3dot) + x1 - x3)) + (n-1)*(n+2)*np.multiply(x3,y3))
    dL_drho = np.multiply((n+1)*sint, np.multiply(y5,(x1-(n*x3))) + np.multiply(x5,(y1-(n*y3)))) - \
        np.multiply(np.multiply(g_norm,sint),(np.multiply(x1,Y) + np.multiply(y1,X))) - \
        np.multiply((np.power(sint,2.)),np.multiply(y1,x5dot)) - \
        (n+1)*np.multiply(sint,np.multiply(y1,x5)) - \
        np.multiply((np.power(sint,2.)),np.multiply(x1,y5dot)) - \
        (n+1)*np.multiply(sint,np.multiply(x1,y5)) + \
        np.multiply(8.*piG*np.power(sint,2.),np.multiply(x1,np.multiply(y1,rho_norm)))

    # Return Lagrangian Derivatives
    return dL_dK, dL_dmu, dL_drho

