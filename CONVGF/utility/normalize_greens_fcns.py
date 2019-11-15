# *********************************************************************
# FUNCTION TO NORMALIZE GREEN'S FUNCTIONS ACCORDING TO THE FARRELL (1972) CONVENTION
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

def main(theta,u,v,rad):

    # rad = radius of body
    
    # Convert Theta to Radians
    theta_rad = np.multiply(theta,(pi/180.))

    # Normalize Displacement Greens Functions
    # According to Agnew (2012) Convention (and Farrell 1972)
    unorm = (rad**2) * u * (2*np.sin(theta_rad/2.))
    vnorm = (rad**2) * v * (2*np.sin(theta_rad/2.))

    # Return Normalized Displacement Greens Fcns
    return unorm,vnorm

