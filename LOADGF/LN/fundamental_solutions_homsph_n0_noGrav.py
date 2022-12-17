# *********************************************************************
# FUNCTION TO COMPUTE ANALYTICAL SOLUTIONS FOR SPHEROIDAL DEFORMATION 
# OF A HOMOGENEOUS ELASTIC SPHERE
# SPECIAL CASE OF NO GRAVITY and N=0
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

    # Modified the homogeneous-sphere starting solutions 
    # based on Takeuchi & Saito (1972) eqs. 104 and 105; note
    # that the equation for k (eq. 99) reduces to (omega/alpha)
    # or (omega/beta), depending on +/- sign, when gamma is set to zero. 

    # Compute Additional Non-Dimensional Parameters
    n1 = n*(n+1.)
    vp = np.sqrt((lndi+2.*mndi)/rndi)
    vs = np.sqrt(mndi/rndi)
    x_alpha = (wnd/vp)*si
    x_beta = (wnd/vs)*si
    x1sqr = x_alpha**2.
    x2sqr = x_beta**2.

    # Compute Bessel Function Parameter Expansions
    # See Takeuchi & Saito (1972), EQ. 102
    # See also Abromowitz & Stegun, PG. 437, EQ. 10.1.2
    ## Compute the Bessel functions using expansion formulas (Takeuchi & Saito 1972, Eq. 102)
    phi1_n = 1. - x1sqr/(2.*(2.*n+3.)) + (x1sqr**2)/(4.*(2.*n+3.)*(2.*n+5.)*2.) - \
        (x1sqr**3)/(math.factorial(3)*(2.**3)*(2.*n+7.)*(2.*n+5.)*(2.*n+3.)) + \
        (x1sqr**4)/(math.factorial(4)*(2.**4)*(2.*n+9.)*(2.*n+7.)*(2.*n+5.)*(2.*n+3.)) - \
        (x1sqr**5)/(math.factorial(5)*(2.**5)*(2.*n+11.)*(2.*n+9.)*(2.*n+7.)*(2.*n+5.)*(2.*n+3.)) + \
        (x1sqr**6)/(math.factorial(6)*(2.**6)*(2.*n+13.)*(2.*n+11.)*(2.*n+9.)*(2.*n+7.)*(2.*n+5.)*(2.*n+3.)) - \
        (x1sqr**7)/(math.factorial(7)*(2.**7)*(2.*n+15.)*(2.*n+13.)*(2.*n+11.)*(2.*n+9.)*(2.*n+7.)*(2.*n+5.)*(2.*n+3.))
    psi1_n = (1.-phi1_n)*((2.*(2.*n+3.))/x1sqr)
    phi1_np1 = 1. - x1sqr/(2.*(2.*(n+1.)+3.)) + (x1sqr**2)/(4.*(2.*(n+1.)+3.)*(2.*(n+1.)+5.)*2.) - \
        (x1sqr**3)/(math.factorial(3)*(2.**3)*(2.*(n+1.)+7.)*(2.*(n+1.)+5.)*(2.*(n+1.)+3.)) + \
        (x1sqr**4)/(math.factorial(4)*(2.**4)*(2.*(n+1.)+9.)*(2.*(n+1.)+7.)*(2.*(n+1.)+5.)*(2.*(n+1.)+3.)) - \
        (x1sqr**5)/(math.factorial(5)*(2.**5)*(2.*(n+1.)+11.)*(2.*(n+1.)+9.)*(2.*(n+1.)+7.)*(2.*(n+1.)+5.)*(2.*(n+1.)+3.)) + \
        (x1sqr**6)/(math.factorial(6)*(2.**6)*(2.*(n+1.)+13.)*(2.*(n+1.)+11.)*(2.*(n+1.)+9.)*(2.*(n+1.)+7.)*(2.*(n+1.)+5.)*(2.*(n+1.)+3.)) - \
        (x1sqr**7)/(math.factorial(7)*(2.**7)*(2.*(n+1.)+15.)*(2.*(n+1.)+13.)*(2.*(n+1.)+11.)*(2.*(n+1.)+9.)*(2.*(n+1.)+7.)*(2.*(n+1.)+5.)*(2.*(n+1.)+3.))

    # A SOLUTION (P-wave for n=0 mode; no shear motion)
    # Set gamma = 0 in equation for k^2 in Eq. 99 -> k^2 = w^2 / alpha^2
    # Inferred by comparing the differences between Eqs. 98 and 104 in Takeuchi & Saito (1972)
    # Then, used the same pattern to adapt Eq. 102 to no-gravity case
    # Pattern: f = h = 1; Ignore any terms without f or h
    f1 = 1.
    h1 = 1.
    Y11 = -(si**(n+1.)/(2.*n+3.))*(0.5*n*h1*psi1_n + f1*phi1_np1)
    Y21 = -(lndi+2.*mndi)*(si**n)*f1*phi1_n + \
        (mndi*(si**n)/(2.*n+3.))*(-(n*(n-1.))*h1*psi1_n + 2.*(2.*f1)*phi1_np1)

    # Form Starting Solution Array
    Y = np.array([[Y11, Y21]])

    # Return Y-Variable Starting Solutions
    return Y

