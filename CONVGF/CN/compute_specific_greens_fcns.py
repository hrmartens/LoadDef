# *********************************************************************
# FUNCTION TO COMPUTE EAST, NORTH, and UP COMPONENTS OF DISPLACEMENT RESPONSE
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
from math import pi

def main(haz,uint,vint):

    # Vertical Component of Displacement Only Depends on Angular Source-Receiver Distance
    ur = uint.copy()

    # Convert haz to Radians
    haz_rad = np.multiply(haz,(pi/180.))
   
    # Compute East and North Components of Horizontal Displacement Greens Functions
    ue = np.multiply(vint,np.sin(haz_rad))
    un = np.multiply(vint,np.cos(haz_rad))

    # Return Specific (Geographically Tied) Greens Functions for Each Load Point
    return ur,ue,un
 

## Special notes when using a common geographic mesh:
# Per the small-angle approximation, when the width of the cell is small, then the integrals over the horizontal displacement response
#  reduce to [-beta*cos(alpha)] for the north component and [-beta*sin(alpha)] for the east component.         
#  See equations 4.221 and 4.222 in H.R. Martens (2016, Caltech thesis). Here, the T(alpha) function is included in the integration. 
#  When we use a common mesh, the convolution mesh is no longer symmetric about the station; therefore, it does not make sense to 
#  include T(alpha) in the integration. However, we can see that the T(alpha) function can be moved outside the integral when the 
#  value of beta is small (and we can invoke the small-angle approximation): 2*sin(beta/2) reduces to 2*(beta/2) = beta.
#  In this way, we treat the horizontal-component integration in the same way as the vertical component:
#  Integrate over the area of each patch (area element on a sphere): int_theta int_phi r^2 sin(theta) d(theta) d(phi)
#   where theta is co-latitude and phi is azimuth in a geographic coordinate system.
#  Next, we can multiply the horizontal solutions by [-cos(alpha)] and [-sin(alpha)] to convert to north and east components, respectively,
#   where alpha is the azimuthal angle between the north pole and the load point, as subtended by the station, and as measured clockwise from north.

