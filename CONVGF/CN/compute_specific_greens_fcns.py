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

