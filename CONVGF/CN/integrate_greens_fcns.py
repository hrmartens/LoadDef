# *********************************************************************
# FUNCTION TO INTEGRATE DISPLACEMENT GREEN'S FUNCTIONS OVER CELLS OF THE
# TEMPLATE GRID
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
import scipy as sc
from scipy import interpolate
from math import pi

def main(gldel,glazm,ldel,lazm,tck_gfu,tck_gfv):

    # Extract Points from Interpolated Greens Functions
    gfu = interpolate.splev(ldel,tck_gfu,der=0)
    gfv = interpolate.splev(ldel,tck_gfv,der=0)

    # Determine Increments in Delta and Azimuth
    del_incs = np.multiply(ldel - gldel[0:-1], 2.)
    azm_incs = np.multiply(lazm - glazm, 2.)

    # Convert Angles to Radians
    del_incs_rad = np.multiply(del_incs,(pi/180.))
    ldel_rad     = np.multiply(ldel,(pi/180.))
    azm_incs_rad = np.multiply(azm_incs,(pi/180.))

    # Integrate the Greens Functions -- Delta Component
    # See Section 4.1 of Agnew (2012), SPOTL Manual
    # Basically computing the area of a spherical patch (area element on a sphere).
    # The normalization factor for the LGF, which mitigates the singularity at 
    # theta=0 and therefore helps with LGF interpolation, is also included in the
    # integration over each patch. 
    intu = 4. * gfu * np.cos(ldel_rad/2.) * np.sin(del_incs_rad/4.)
    intv = 4. * gfv * np.cos(ldel_rad/2.) * np.sin(del_incs_rad/4.)

    # Integrate the Greens Functions -- Azimuthal Component
    # See, for example, 4.220-4.222 in H.R. Martens (2016, Caltech thesis).
    # Note that (2.*np.sin(azm_incs_rad[0]/2.)) reduces to azm_incs_rad[0] when 
    # the increment is small per the small-angle approximation.
    # Multiplication by the T(alpha) term (to convert to east and north components)
    # happens in a separate function: computing the specific Green's functions.
    intu = intu * azm_incs_rad[0]
    intv = intv * (2.*np.sin(azm_incs_rad[0]/2.))

    # Create Full 1-D Arrays
    xv,yv = sc.meshgrid(intu,lazm)
    uint = xv.flatten()
    xv,yv = sc.meshgrid(intv,lazm)
    vint = xv.flatten()

    # Return Integrated Greens Functions
    return uint,vint

