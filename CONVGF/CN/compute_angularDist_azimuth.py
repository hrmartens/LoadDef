# *********************************************************************
# FUNCTION TO COMPUTE ANGULAR DISTANCE AND AZIMUTH BETWEEN POINTS ON A SPHERE
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
from CONVGF.utility import distance_az_baz

def main(rlat,rlon,ilat,ilon):

    # WGS84 (Table 2.3, Physical Geodesy 2nd Ed., Hofmann-Wellenhof & Moritz)
    flattening = 1./298.257223563
    b_over_a   = 1. - flattening

    # Compute Spherical Distance Angle (delta), Azimuth (az), and Back Azimuth (baz)
    # Note that Coordinates are First Converted from Geographic to Geocentric
    az, baz, delta = distance_az_baz.distaz2(rlat,rlon,ilat,ilon,b_over_a)

    # Horizontal Response Points Radially Away from Load (180-Deg from Azimuth)
    haz = az + 180.

    # Return Delta and Azimuth
    return delta,haz

