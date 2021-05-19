#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO CREATE A GRID OF STATION/GRID-POINT LOCATIONS OVER DESIGNATED LAND AREAS
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

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd() + "/../../")

# IMPORT PYTHON MODULES
import numpy as np
import scipy as sc
from scipy import interpolate
from CONVGF.utility import read_lsmask
from CONVGF.CN import interpolate_lsmask
import matplotlib.pyplot as plt
 
# Define the Grid Spacing
grid_spacing_x = 1.0
grid_spacing_y = 1.0

# Bounds for the Grid
south = 30.
north = 52.
west = 232. # [0,360]
east = 252. # [0,360]

# --------------- BEGIN CODE ----------------- #

# Create Folders
if not (os.path.isdir("../../output/Grid_Files/")):
    os.makedirs("../../output/Grid_Files/")
if not (os.path.isdir("../../output/Grid_Files/LandDB/")):
    os.makedirs("../../output/Grid_Files/LandDB/")
outdir = "../../output/Grid_Files/LandDB/"
 
# Create the Grid
x_range = np.arange(west,east,grid_spacing_x)
y_range = np.arange(south,north,grid_spacing_y)
xv,yv = np.meshgrid(x_range,y_range)
mylon = np.ravel(xv)
mylat = np.ravel(yv)
num = np.arange(0,len(mylat),1)
print(':: Total Number of Locations: %6d' %(len(mylat)))

# Output Grid to File for Plotting in GMT
land_out = "LandOceanGrid_x_" + str(grid_spacing_x) + "_y_" + str(grid_spacing_y) + "_bounds_" + str(int(south)) + "_" + str(int(north)) + "_" + str(int(west)) + "_" + str(int(east)) + ".txt"
land_file = (outdir + land_out)
# Prepare Data
all_land_data = np.column_stack((mylat,mylon,num))
# Write Data to File
np.savetxt(land_file, all_land_data, fmt='%f %f %08d')

# Plot
plt.plot(mylon,mylat,'.',ms=6)
plt.show()

