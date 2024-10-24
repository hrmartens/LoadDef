# *********************************************************************
# FUNCTION TO COMPUTE ANALYTICAL SOLUTIONS FOR SPHEROIDAL DEFORMATION 
# OF A HOMOGENEOUS ELASTIC SPHERE
#
# Copyright (c) 2014-2024: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
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

# Compute Homogeneous Sphere Starting Solutions
# See Takeuchi & Saito (1972)
# Methods in Computational Physics, Advances in 
# Research and Applications (Volume 11: Seismology:
# Surface Waves and Earth Oscillations),
# Ed. Bruce A. Bolt, 1972

import numpy as np
from scipy import interpolate
import math
from scipy.special import spherical_jn as sph_jn

def main(si,n,tck_lnd,tck_mnd,tck_rnd,wnd,piG):

    # Interpolate Parameters at Current Radius
    lndi = interpolate.splev(si,tck_lnd,der=0)
    rndi = interpolate.splev(si,tck_rnd,der=0)
    mndi = interpolate.splev(si,tck_mnd,der=0)

    # Compute Additional Non-Dimensional Parameters
    n1 = n*(n+1.)
    vp = np.sqrt((lndi+2.*mndi)/rndi)
    vs = np.sqrt(mndi/rndi)
    gamma = 4.*piG*rndi/3.
    # k May be Imaginary
    ksq1 = 0.5*((((wnd**2) + 4.*gamma)/(vp**2)) + ((wnd**2)/(vs**2)) - \
        np.sqrt(((wnd**2)/(vs**2) - ((wnd**2) + 4.*gamma)/(vp**2))**2 + \
        (4.*n1*(gamma**2)/((vs**2)*(vp**2)))))
    ksq2 = 0.5*((((wnd**2) + 4.*gamma)/(vp**2)) + ((wnd**2)/(vs**2)) + \
        np.sqrt(((wnd**2)/(vs**2) - ((wnd**2) + 4.*gamma)/(vp**2))**2 + \
        (4.*n1*(gamma**2)/((vs**2)*(vp**2)))))
    # From Takeuchi & Saito (1972), Eq. 99: Factored for Numerical Stability
    f1 = (1./gamma)*(vs*np.lib.scimath.sqrt(ksq1)+wnd)*(vs*np.lib.scimath.sqrt(ksq1)-wnd)
    f2 = (1./gamma)*(vs*np.lib.scimath.sqrt(ksq2)+wnd)*(vs*np.lib.scimath.sqrt(ksq2)-wnd)
    # Imaginary Part is Effectively Zero -- Get Rid of It
    f1 = f1.real
    f2 = f2.real
    h1 = f1 - (n+1.)
    h2 = f2 - (n+1.)
    # x May be Imaginary -- Only Even Powers will be Used Later
    x1 = np.lib.scimath.sqrt(ksq1)*si
    x2 = np.lib.scimath.sqrt(ksq2)*si

    # Compute the Squares
    x1sqr = x1*x1
    x1sqr = x1sqr.real
    x2sqr = x2*x2
    x2sqr = x2sqr.real

    # Compute Bessel Function Parameter Expansions
    # See Takeuchi & Saito (1972), EQ. 102
    # See also Abromowitz & Stegun, PG. 437, EQ. 10.1.2
    ## Compute the Bessel functions using built-in functions (less stable than the expansions!)
    #phi1_n = factorial2(int(2*n+1))*sph_jn(int(n),x1sqr)[0][int(n)]/pow(x1sqr,n)
    #psi1_n = (1.-phi1_n)*((2.*(2.*n+3.))/x1sqr)
    #phi2_n = factorial2(int(2*n+1))*sph_jn(int(n),x2sqr)[0][int(n)]/pow(x2sqr,n)
    #psi2_n = (1.-phi2_n)*((2.*(2.*n+3.))/x2sqr)
    #phi1_np1 = factorial2(int(2*(n+1)+1))*sph_jn(int(n+1),x1sqr)[0][(int(n)+1)]/pow(x1sqr,(n+1))
    #phi2_np1 = factorial2(int(2*(n+1)+1))*sph_jn(int(n+1),x2sqr)[0][(int(n)+1)]/pow(x2sqr,(n+1))
    ## Compute the Bessel functions using expansion formulas (Takeuchi & Saito 1972, Eq. 102)
    phi1_n = 1. - x1sqr/(2.*(2.*n+3.)) + (x1sqr**2)/(4.*(2.*n+3.)*(2.*n+5.)*2.) - \
        (x1sqr**3)/(math.factorial(3)*(2.**3)*(2.*n+7.)*(2.*n+5.)*(2.*n+3.)) + \
        (x1sqr**4)/(math.factorial(4)*(2.**4)*(2.*n+9.)*(2.*n+7.)*(2.*n+5.)*(2.*n+3.)) - \
        (x1sqr**5)/(math.factorial(5)*(2.**5)*(2.*n+11.)*(2.*n+9.)*(2.*n+7.)*(2.*n+5.)*(2.*n+3.)) + \
        (x1sqr**6)/(math.factorial(6)*(2.**6)*(2.*n+13.)*(2.*n+11.)*(2.*n+9.)*(2.*n+7.)*(2.*n+5.)*(2.*n+3.)) - \
        (x1sqr**7)/(math.factorial(7)*(2.**7)*(2.*n+15.)*(2.*n+13.)*(2.*n+11.)*(2.*n+9.)*(2.*n+7.)*(2.*n+5.)*(2.*n+3.))
    psi1_n = (1.-phi1_n)*((2.*(2.*n+3.))/x1sqr)
    phi2_n = 1. - x2sqr/(2.*(2.*n+3.)) + (x2sqr**2)/(4.*(2.*n+3.)*(2.*n+5.)*2.) - \
        (x2sqr**3)/(math.factorial(3)*(2.**3)*(2.*n+7.)*(2.*n+5.)*(2.*n+3.)) + \
        (x2sqr**4)/(math.factorial(4)*(2.**4)*(2.*n+9.)*(2.*n+7.)*(2.*n+5.)*(2.*n+3.)) - \
        (x2sqr**5)/(math.factorial(5)*(2.**5)*(2.*n+11.)*(2.*n+9.)*(2.*n+7.)*(2.*n+5.)*(2.*n+3.)) + \
        (x2sqr**6)/(math.factorial(6)*(2.**6)*(2.*n+13.)*(2.*n+11.)*(2.*n+9.)*(2.*n+7.)*(2.*n+5.)*(2.*n+3.)) - \
        (x2sqr**7)/(math.factorial(7)*(2.**7)*(2.*n+15.)*(2.*n+13.)*(2.*n+11.)*(2.*n+9.)*(2.*n+7.)*(2.*n+5.)*(2.*n+3.))
    psi2_n = (1.-phi2_n)*((2.*(2.*n+3.))/x2sqr)

    phi1_np1 = 1. - x1sqr/(2.*(2.*(n+1.)+3.)) + (x1sqr**2)/(4.*(2.*(n+1.)+3.)*(2.*(n+1.)+5.)*2.) - \
        (x1sqr**3)/(math.factorial(3)*(2.**3)*(2.*(n+1.)+7.)*(2.*(n+1.)+5.)*(2.*(n+1.)+3.)) + \
        (x1sqr**4)/(math.factorial(4)*(2.**4)*(2.*(n+1.)+9.)*(2.*(n+1.)+7.)*(2.*(n+1.)+5.)*(2.*(n+1.)+3.)) - \
        (x1sqr**5)/(math.factorial(5)*(2.**5)*(2.*(n+1.)+11.)*(2.*(n+1.)+9.)*(2.*(n+1.)+7.)*(2.*(n+1.)+5.)*(2.*(n+1.)+3.)) + \
        (x1sqr**6)/(math.factorial(6)*(2.**6)*(2.*(n+1.)+13.)*(2.*(n+1.)+11.)*(2.*(n+1.)+9.)*(2.*(n+1.)+7.)*(2.*(n+1.)+5.)*(2.*(n+1.)+3.)) - \
        (x1sqr**7)/(math.factorial(7)*(2.**7)*(2.*(n+1.)+15.)*(2.*(n+1.)+13.)*(2.*(n+1.)+11.)*(2.*(n+1.)+9.)*(2.*(n+1.)+7.)*(2.*(n+1.)+5.)*(2.*(n+1.)+3.))
    phi2_np1 = 1. - x2sqr/(2.*(2.*(n+1.)+3.)) + (x2sqr**2)/(4.*(2.*(n+1.)+3.)*(2.*(n+1.)+5.)*2.) - \
        (x2sqr**3)/(math.factorial(3)*(2.**3)*(2.*(n+1.)+7.)*(2.*(n+1.)+5.)*(2.*(n+1.)+3.)) + \
        (x2sqr**4)/(math.factorial(4)*(2.**4)*(2.*(n+1.)+9.)*(2.*(n+1.)+7.)*(2.*(n+1.)+5.)*(2.*(n+1.)+3.)) - \
        (x2sqr**5)/(math.factorial(5)*(2.**5)*(2.*(n+1.)+11.)*(2.*(n+1.)+9.)*(2.*(n+1.)+7.)*(2.*(n+1.)+5.)*(2.*(n+1.)+3.)) + \
        (x2sqr**6)/(math.factorial(6)*(2.**6)*(2.*(n+1.)+13.)*(2.*(n+1.)+11.)*(2.*(n+1.)+9.)*(2.*(n+1.)+7.)*(2.*(n+1.)+5.)*(2.*(n+1.)+3.)) - \
        (x2sqr**7)/(math.factorial(7)*(2.**7)*(2.*(n+1.)+15.)*(2.*(n+1.)+13.)*(2.*(n+1.)+11.)*(2.*(n+1.)+9.)*(2.*(n+1.)+7.)*(2.*(n+1.)+5.)*(2.*(n+1.)+3.))

    # FIRST SOLUTION
    Y11 = -(si**(n+1.)/(2.*n+3.))*(0.5*n*h1*psi1_n + f1*phi1_np1)
    Y21 = -(lndi+2.*mndi)*(si**n)*f1*phi1_n + \
	(mndi*(si**n)/(2.*n+3.))*(-(n*(n-1.))*h1*psi1_n + 2.*(2.*f1+n1)*phi1_np1)
    Y51 = (si**(n+2.))*(((vp**2)*f1 - (n+1.)*(vs**2))/(si**2) - \
	((3.*gamma*f1)/(2.*(2.*n+3.)))*psi1_n)
    Y61 = (2.*n+1.)*(si**(n+1.))*(((vp**2)*f1 - (n+1.)*(vs**2))/(si**2) - \
	((3.*gamma*f1)/(2.*(2.*n+3.)))*psi1_n) + \
	((3.*n*gamma*h1*(si**(n+1.)))/(2.*(2.*n+3.)))*psi1_n

    # SECOND SOLUTION
    Y12 = -(si**(n+1.)/(2.*n+3.))*(0.5*n*h2*psi2_n + f2*phi2_np1)
    Y22 = -(lndi+2.*mndi)*(si**n)*f2*phi2_n + \
        (mndi*(si**n)/(2.*n+3.))*(-(n*(n-1.))*h2*psi2_n + 2.*(2.*f2+n1)*phi2_np1)
    Y52 = (si**(n+2.))*(((vp**2)*f2 - (n+1.)*(vs**2))/(si**2) - \
        ((3.*gamma*f2)/(2.*(2.*n+3.)))*psi2_n)
    Y62 = (2.*n+1.)*(si**(n+1.))*(((vp**2)*f2 - (n+1.)*(vs**2))/(si**2) - \
        ((3.*gamma*f2)/(2.*(2.*n+3.)))*psi2_n) + \
        ((3.*n*gamma*h2*(si**(n+1.)))/(2.*(2.*n+3.)))*psi2_n

    # CONVERT TAKEUCHI & SAITO Y CONVENTION BACK TO SMYLIE CONVENTION
    Y61 = Y61 - ((n+1.)/si)*Y51
    Y62 = Y62 - ((n+1.)/si)*Y52

    # Form Starting Solution Array
    Y = np.array([[Y11, Y21, Y51, Y61], \
	[Y12, Y22, Y52, Y62]])

    # Return Y-Variable Starting Solutions
    return Y

