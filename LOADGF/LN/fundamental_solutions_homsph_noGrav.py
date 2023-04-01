# *********************************************************************
# FUNCTION TO COMPUTE ANALYTICAL SOLUTIONS FOR SPHEROIDAL DEFORMATION 
# OF A HOMOGENEOUS ELASTIC SPHERE
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

    # FIRST SOLUTION
    # Set gamma = 0 in equation for k^2 in Eq. 99 -> k^2 = w^2 / alpha^2
    # Factor out a common factor of sqrt(pi/4) * (x/2)^n * (1/gamma(n+(3/2)))
    # See Abromowitz & Stegun Eqs. 9.1.7, 10.1.1, 6.1.15, 6.1.16, 6.1.12 (must do some re-arranging)
    jn_xalpha = 1.
    jnp1_xalpha = (x_alpha/2.) * (1. / (n + (3./2.)))
    Y11 = (1./si) * (n * jn_xalpha - x_alpha * jnp1_xalpha)
    Y21 = (1./(si**2.)) * (-(lndi+(2.*mndi))*(x_alpha**2.)*jn_xalpha + (2.*mndi)*((n*(n-1.)*jn_xalpha)+(2.*x_alpha*jnp1_xalpha)))
    Y31 = (1./si) * (jn_xalpha)
    Y41 = (1./(si**2.)) * (2.*mndi) * ((n-1.)*jn_xalpha - x_alpha*jnp1_xalpha)

    # SECOND SOLUTION
    # Set gamma = 0 in equation for k^2 in Eq. 99 -> k^2 = w^2 / beta^2
    # Factor out a common factor of sqrt(pi/4) * (x/2)^n * (1/gamma(n+(3/2)))
    # See Abromowitz & Stegun Eqs. 9.1.7, 10.1.1, 6.1.15, 6.1.16, 6.1.12 (must do some re-arranging)
    jn_xbeta = 1.
    jnp1_xbeta = (x_beta/2.) * (1. / (n + (3./2.)))
    Y12 = (1./si) * (-n*(n+1)*jn_xbeta)
    Y22 = (1./(si**2.)) * (2.*mndi) * (-n*((n**2.)-1.)*jn_xbeta + n*(n+1.)*x_beta*jnp1_xbeta)
    Y32 = (1./si) * (-(n+1.)*jn_xbeta + x_beta*jnp1_xbeta)
    Y42 = (1./(si**2.)) * (mndi) * ((x_beta**2.)*jn_xbeta - 2.*((n**2.)-1.)*jn_xbeta - 2.*x_beta*jnp1_xbeta)

    # Form Starting Solution Array
    Y = np.array([[Y11, Y21, Y31, Y41], \
	[Y12, Y22, Y32, Y42]])

    # Return Y-Variable Starting Solutions
    return Y

