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
grid_spacing_x = 0.2
grid_spacing_y = 0.2

# Bounds for the Grid
south = 63.
north = 67.
west = 334. # [0,360]
east = 348. # [0,360]

# Land-Sea Mask
land_sea = ("../../input/Land_Sea/ETOPO1_Ice_g_gmt4_wADD.txt")

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
xv = np.ravel(xv)
yv = np.ravel(yv)

# Read In the Land-Sea Mask 
print(':: Reading in the Land-Sea Mask.')
lslat,lslon,lsmask = read_lsmask.main(land_sea) 

# Ensure that Land-Sea Mask Longitudes are in Range 0-360
neglon_idx = np.where(lslon<0.)
lslon[neglon_idx] = lslon[neglon_idx] + 360.
 
# Determine the Land-Sea Mask (1' Resolution) From ETOPO1 (and Optionally GSHHG as well)
print(':: Interpolating Land-Sea Mask onto Grid.')
lsmk = interpolate_lsmask.main(yv,xv,lslat,lslon,lsmask)

# Apply Land-Sea Mask
print(':: Applying Land-Sea Mask to the Grid.')
mylat = yv[lsmk == 1]
mylon = xv[lsmk == 1]
num = np.arange(0,len(mylat),1)
print(':: Total Number of Locations: %6d' %(len(mylat)))

# Output Grid to File for Plotting in GMT
land_out = "LandGrid_x_" + str(grid_spacing_x) + "_y_" + str(grid_spacing_y) + "_bounds_" + str(int(south)) + "_" + str(int(north)) + "_" + str(int(west)) + "_" + str(int(east)) + ".txt"
land_file = (outdir + land_out)
# Prepare Data
all_land_data = np.column_stack((mylat,mylon,num))
# Write Data to File
#f_handle = open(land_file, 'w')
#np.savetxt(f_handle, all_land_data, fmt='%f %f %08d')
#f_handle.close()
np.savetxt(land_file, all_land_data, fmt='%f %f %08d')

# Plot
plt.plot(mylon,mylat,'.',ms=6)
plt.show()

