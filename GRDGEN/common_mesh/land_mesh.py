#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO CREATE A MESH OF GEOGRAPHIC GRIDLINES and MIDPOINTS 
#  OVER DESIGNATED LAND AREAS
# 
# Copyright (c) 2014-2022: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
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
from math import pi
import netCDF4 
from CONVGF.utility import read_lsmask
from CONVGF.CN import interpolate_lsmask
import matplotlib.pyplot as plt
 
# --------------- SPECIFY USER INPUTS --------------------- #

# 1. Specify the region of interest
slat = 28.
nlat = 50.
wlon = 233. # [0,360]
elon = 258. # [0,360]

# 2. Specify the mesh resolution (in degrees)
grid_spacing_x = 0.01
grid_spacing_y = 0.01
 
# 3. Land-Sea Mask
land_sea = ("../../input/Land_Sea/ETOPO1_Ice_g_gmt4_wADD.txt")

# 4. Write Load Information to a netCDF-formatted File? (Default for convolution)
write_nc = True

# 5. Write Load Information to a Text File? (Alternative for convolution)
write_txt = True

# ------------------ END USER INPUTS ----------------------- #

# -------------------- BEGIN CODE -------------------------- #

# Create Folders
if not (os.path.isdir("../../output/Grid_Files/")):
    os.makedirs("../../output/Grid_Files/")
if not (os.path.isdir("../../output/Grid_Files/nc/")):
    os.makedirs("../../output/Grid_Files/nc/")
if not (os.path.isdir("../../output/Grid_Files/nc/commonMesh/")):
    os.makedirs("../../output/Grid_Files/nc/commonMesh/")
if not (os.path.isdir("../../output/Grid_Files/text/")):
    os.makedirs("../../output/Grid_Files/text/")
if not (os.path.isdir("../../output/Grid_Files/text/commonMesh/")):
    os.makedirs("../../output/Grid_Files/text/commonMesh/")
 
# Determine Cell Grid Lines
inumy = int(((nlat-slat)/grid_spacing_y)+1)         # number of latitude increments
gllat = np.linspace(slat,nlat,num=inumy)            # lines of latitude
inumx = int(((elon-wlon)/grid_spacing_x)+1)         # number of longitude increments
gllon = np.linspace(wlon,elon,num=inumx)            # lines of longitude
if ((elon-wlon) == 360.):
    gllon = gllon[0:-1]                             # if using full Earth, then don't include last element (same as first element)    

# Determine Cell Midpoints
lon_mdpts = gllon[0:-1] + grid_spacing_x/2.         # midpoints between meridional gridlines
lat_mdpts = gllat[0:-1] + grid_spacing_y/2.         # midpoints between latitudinal gridlines
  
# Determine Unit Area of Each Cell
# Note: Assumes equal azimuthal (i.e., meridional) spacing everywhere.
# Equation is for the area of a spherical patch (i.e., area element on surface of a sphere).
#  int_theta int_phi (r^2 sin(theta) d(theta) d(phi))
#  Theta is co-latitude. Phi is azimuth (longitude).
#  The result of the integration is: r^2 * (phi2 - phi1) * (cos[theta1] - cos[theta2])
#  Also note that: cos(co-latitude) = sin(latitude)  
# A good check is to ensure that the area of the sphere comes out to 4*pi*r^2 when integrating over the entire surface. 
#  This is phi going from 0 to 2 pi and theta going from 0 to pi. We have:
#  r^2 * 2 * pi * (1 - -1) = r^2 * 2 * pi * 2 = 4 * pi * r^2.
# For a unit sphere, r=1.
unit_area = []
gllat_rad = np.multiply(gllat,(pi/180.))
lon_inc_rad = np.multiply(grid_spacing_x,(pi/180.))
for ii in range(1,len(gllat_rad)):
    unit_area.append(np.multiply(lon_inc_rad,\
        np.sin(gllat_rad[ii])-np.sin(gllat_rad[ii-1])))
unit_area = np.asarray(unit_area)

# Create the Grid
xv1,yv1 = np.meshgrid(lon_mdpts,lat_mdpts)
xv2,yv2 = np.meshgrid(lon_mdpts,unit_area)

# Read In the Land-Sea Mask 
print(':: Reading in the Land-Sea Mask.')
lslat,lslon,lsmask = read_lsmask.main(land_sea) 

# Ensure that Land-Sea Mask Longitudes are in Range 0-360
neglon_idx = np.where(lslon<0.)
lslon[neglon_idx] = lslon[neglon_idx] + 360.
 
# Determine the Land-Sea Mask (1' Resolution) From ETOPO1 (and Optionally GSHHG as well)
print(':: Interpolating Land-Sea Mask onto Grid.')
lsmk = interpolate_lsmask.main(yv1,xv1,lslat,lslon,lsmask)

# Apply Land-Sea Mask
print(':: Applying Land-Sea Mask to the Grids.')
landlat = yv1[lsmk == 1]
landlon = xv1[lsmk == 1]
unit_area = yv2[lsmk == 1]
print(':: Total Number of Locations: %6d' %(len(landlat)))

# Output Load Cells to File for Use with LoadDef
if (write_nc == True):
    print(":: Writing netCDF-formatted file.")
    outname = ("commonMesh_" + str(slat) + "_" + str(nlat) + "_" + str(wlon) + "_" + str(elon) + "_" + str(grid_spacing_y) + "_" + str(grid_spacing_x) + ".nc")
    outfile = ("../../output/Grid_Files/nc/commonMesh/" + outname)
    # Open new NetCDF file in "write" mode
    dataset = netCDF4.Dataset(outfile,'w',format='NETCDF4_CLASSIC')
    # Define dimensions for variables
    numpts = len(landlat)
    gridline_lat = dataset.createDimension('gridline_lat',len(gllat))
    gridline_lon = dataset.createDimension('gridline_lon',len(gllon))
    midpoint_lat = dataset.createDimension('midpoint_lat',numpts)
    midpoint_lon = dataset.createDimension('midpoint_lon',numpts)
    unit_area_patch = dataset.createDimension('unit_area_patch',numpts)
    # Create variables 
    gridline_lats = dataset.createVariable('gridline_lat',float,('gridline_lat',))
    gridline_lons = dataset.createVariable('gridline_lon',float,('gridline_lon',))
    midpoint_lats = dataset.createVariable('midpoint_lat',float,('midpoint_lat',))
    midpoint_lons = dataset.createVariable('midpoint_lon',float,('midpoint_lon',))
    unit_area_patches = dataset.createVariable('unit_area_patch',float,('unit_area_patch',))
    # Add units
    gridline_lats.units = 'degree_north'
    gridline_lons.units = 'degree_east'
    midpoint_lats.units = 'degree_north'
    midpoint_lons.units = 'degree_east'
    unit_area_patches.units = 'dimensionless (need to multiply by r^2 when used)'
    # Assign data
    gridline_lats[:] = gllat
    gridline_lons[:] = gllon
    midpoint_lats[:] = landlat
    midpoint_lons[:] = landlon
    unit_area_patches[:] = unit_area
    # Write Data to File
    dataset.close()
if (write_txt == True):
    print(":: Writing plain-text file.")
    outname = ("commonMesh_" + str(slat) + "_" + str(nlat) + "_" + str(wlon) + "_" + str(elon) + "_" + str(grid_spacing_y) + "_" + str(grid_spacing_x) + ".txt")
    outfile = ("../../output/Grid_Files/text/commonMesh/" + outname)
    # Prepare Data
    all_data = np.array(list(zip(landlat,landlon,unit_area)), dtype=[('landlat',float),('landlon',float),('unit_area',float)])
    # Write Data to File
    np.savetxt(outfile, all_data, fmt=["%.15f",]*3, delimiter="      ")

# Plot
plt.plot(landlon,landlat,'.',ms=6)
plt.show()


