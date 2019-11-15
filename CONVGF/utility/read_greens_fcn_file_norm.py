# *********************************************************************
# FUNCTION TO READ IN A FILE CONTAINING DISPLACEMENT LOAD GREEN'S FUNCTIONS
# NORMALIZED ACCORDING TO THE FARRELL (1972) CONVENTION
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

def main(filename,rad):

    # 'rad' = Earth Radius

    # Read Normalized Greens Functions from File
    theta,unorm,vnorm = np.loadtxt(filename,usecols=(0,1,2),unpack=True,skiprows=1)
    
    # Convert Theta to Radians
    theta_rad = np.multiply(theta,(pi/180.))

    # Un-normalize the Displacement Greens Functions (re: Farrell 1972 convention)
    K = 10.**12.
    u = np.divide(unorm, np.multiply(K*rad,theta_rad))
    v = np.divide(vnorm, np.multiply(K*rad,theta_rad))

    return theta,u,v,unorm,vnorm

