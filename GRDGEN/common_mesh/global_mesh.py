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

# 1. Specify the basic mesh resolution (in degrees)
gspace_lon = 1.0 # longitude
gspace_lat = 1.0 # latitude

# 2. Specify bounding box for a region with enhanced mesh resolution (e.g. boundingbox.klokantech.com) | Middle region
#  :: In general, the longitude range should be [0,360]
#  :: In the special case that the bounding box crosses the prime meridian,
#     the range should be [-180,0] for wlon and [0,180] for elon
wlon_mid=213. # range [0,360] | If Bounding Box Crosses Prime Meridian, range = [-180,0]
elon_mid=278. # range [0,360] | If Bounding Box Crosses Prime Meridian, range = [0,180]
slat_mid=18.  # range [-90,90]
nlat_mid=60.  # range [-90,90]

# 3. Specify the enhanced mesh resolution (in degrees) | Middle region
enhanced_lon_mid = 0.1
enhanced_lat_mid = 0.1
 
# 4. Specify bounding box for a region with enhanced mesh resolution (e.g. boundingbox.klokantech.com) | Inner region
#  :: In general, the longitude range should be [0,360]
#  :: In the special case that the bounding box crosses the prime meridian,
#     the range should be [-180,0] for wlon and [0,180] for elon
wlon_inn=233. # range [0,360] | If Bounding Box Crosses Prime Meridian, range = [-180,0]
elon_inn=258. # range [0,360] | If Bounding Box Crosses Prime Meridian, range = [0,180]
slat_inn=28.  # range [-90,90]
nlat_inn=50.  # range [-90,90]

# 5. Specify the enhanced mesh resolution (in degrees) | Inner region
enhanced_lon_inn = 0.01
enhanced_lat_inn = 0.01

# 6. Apply Prime-Meridian Correction? 
#  :: Set to "True" if the Bounding Box Stradles the Prime Meridian
#  :: Ranges Must be [-180,0] and [0,180]
pm_correct = False
 
# 7. Land-Sea Mask
#  :: 0 = do not mask ocean or land (retain full model); 1 = mask out land (retain ocean); 2 = mask out oceans (retain land)
#  :: Recommended: 1 for oceanic; 2 for atmospheric
lsmask_type = 1
land_sea = ("../../input/Land_Sea/ETOPO1_Ice_g_gmt4_wADD.txt")

# 8. Write Load Information to a netCDF-formatted File? (Default for convolution)
write_nc = True

# 9. Write Load Information to a Text File? (Alternative for convolution)
write_txt = False

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
inumy = int((180./gspace_lat)+1)                    # number of latitude increments
gllat1 = np.linspace(-90.,90.,num=inumy)            # lines of latitude
inumx = int((360./gspace_lon)+1)                    # number of longitude increments
gllon1 = np.linspace(0.,360.,num=inumx)             # lines of longitude

# Determine Cell Midpoints
lat_mdpts1 = gllat1[0:-1] + gspace_lat/2.         # midpoints between latitudinal gridlines
lon_mdpts1 = gllon1[0:-1] + gspace_lon/2.         # midpoints between meridional gridlines

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
ua1 = []
gllat1_rad = np.multiply(gllat1,(pi/180.))
lon_inc_rad = np.multiply(gspace_lon,(pi/180.))
for ii in range(1,len(gllat1_rad)):
    ua1.append(np.multiply(lon_inc_rad,\
        np.sin(gllat1_rad[ii])-np.sin(gllat1_rad[ii-1])))
ua1 = np.asarray(ua1)

# Create the Grids
xv1,yv1 = np.meshgrid(lon_mdpts1,lat_mdpts1)
xv2,yv2 = np.meshgrid(lon_mdpts1,ua1)
llon1 = np.ravel(xv1)
llat1 = np.ravel(yv1)
ua1 = np.ravel(yv2)

# If Necessary, Apply Prime-Meridian Correction (Shift to Range [-180,180])
if (pm_correct == True):
    print(':: Applying the prime-meridian correction.')
    if wlon > 0.:
        sys.exit('Error: When applying the prime-meridian correction, the longitudes of the bounding box must range from [-180,180].')
    if elon < 0.:
        sys.exit('Error: When applying the prime-meridian correction, the longitudes of the bounding box must range from [-180,180].')
    pm_correction = np.where(llon1>=180.); pm_correction = pm_correction[0]
    llon1[pm_correction] -= 360.

# Find indices inside bounding box | Middle region
# Delete them from the existing mesh
bboxinside = np.where((llon1 >= wlon_mid) & (llat1 >= slat_mid) & (llon1 <= elon_mid) & (llat1 <= nlat_mid)); bboxinside = bboxinside[0]
llon1 = np.delete(llon1,bboxinside)
llat1 = np.delete(llat1,bboxinside)
ua1 = np.delete(ua1,bboxinside)

# Determine Cell Grid Lines: Enhanced Region | Middle
inumy = int(((nlat_mid-slat_mid)/enhanced_lat_mid)+1)           # number of latitude increments
gllat2 = np.linspace(slat_mid,nlat_mid,num=inumy)               # lines of latitude
inumx = int(((elon_mid-wlon_mid)/enhanced_lon_mid)+1)           # number of longitude increments
gllon2 = np.linspace(wlon_mid,elon_mid,num=inumx)               # lines of longitude

# Determine Cell Midpoints: Enhanced Region | Middle
lat_mdpts2 = gllat2[0:-1] + enhanced_lat_mid/2.         # midpoints between latitudinal gridlines
lon_mdpts2 = gllon2[0:-1] + enhanced_lon_mid/2.         # midpoints between meridional gridlines

# Determine Unit Area of Each Cell: Enhanced Region | Middle
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
ua2 = []
gllat2_rad = np.multiply(gllat2,(pi/180.))
lon_inc_rad = np.multiply(enhanced_lon_mid,(pi/180.))
for ii in range(1,len(gllat2_rad)):
    ua2.append(np.multiply(lon_inc_rad,\
        np.sin(gllat2_rad[ii])-np.sin(gllat2_rad[ii-1])))
ua2 = np.asarray(ua2)
 
# Create the Grids | Middle
xv1,yv1 = np.meshgrid(lon_mdpts2,lat_mdpts2)
xv2,yv2 = np.meshgrid(lon_mdpts2,ua2)
llon2 = np.ravel(xv1)
llat2 = np.ravel(yv1)
ua2 = np.ravel(yv2)

# Concatenate basic and enhanced (middle) grids
llat = np.concatenate([llat1,llat2])
llon = np.concatenate([llon1,llon2])
unit_area = np.concatenate([ua1,ua2])

# Plot
#plt.plot(llon,llat,'.',ms=6)
#plt.show()

# Find indices inside bounding box | Inner region
# Delete them from the existing mesh
bboxinside = np.where((llon >= wlon_inn) & (llat >= slat_inn) & (llon <= elon_inn) & (llat <= nlat_inn)); bboxinside = bboxinside[0]
llon1 = np.delete(llon,bboxinside)
llat1 = np.delete(llat,bboxinside)
ua1 = np.delete(unit_area,bboxinside)

# Determine Cell Grid Lines: Enhanced Region | Inner
inumy = int(((nlat_inn-slat_inn)/enhanced_lat_inn)+1)           # number of latitude increments
gllat2 = np.linspace(slat_inn,nlat_inn,num=inumy)               # lines of latitude
inumx = int(((elon_inn-wlon_inn)/enhanced_lon_inn)+1)           # number of longitude increments
gllon2 = np.linspace(wlon_inn,elon_inn,num=inumx)               # lines of longitude

# Determine Cell Midpoints: Enhanced Region | Inner
lat_mdpts2 = gllat2[0:-1] + enhanced_lat_inn/2.         # midpoints between latitudinal gridlines
lon_mdpts2 = gllon2[0:-1] + enhanced_lon_inn/2.         # midpoints between meridional gridlines

# Determine Unit Area of Each Cell: Enhanced Region | Inner
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
ua2 = []
gllat2_rad = np.multiply(gllat2,(pi/180.))
lon_inc_rad = np.multiply(enhanced_lon_inn,(pi/180.))
for ii in range(1,len(gllat2_rad)):
    ua2.append(np.multiply(lon_inc_rad,\
        np.sin(gllat2_rad[ii])-np.sin(gllat2_rad[ii-1])))
ua2 = np.asarray(ua2)
 
# Create the Grids
xv1,yv1 = np.meshgrid(lon_mdpts2,lat_mdpts2)
xv2,yv2 = np.meshgrid(lon_mdpts2,ua2)
llon2 = np.ravel(xv1)
llat2 = np.ravel(yv1)
ua2 = np.ravel(yv2)

# Concatenate basic and enhanced grids
llat = np.concatenate([llat1,llat2])
llon = np.concatenate([llon1,llon2])
unit_area = np.concatenate([ua1,ua2])

# Plot
#plt.plot(llon,llat,'.',ms=6)
#plt.show()

# If Necessary, Shift Longitude Values back to Original Range ([0,360])
if (pm_correct == True):
    llon[pm_correction] += 360.

# Apply a land-sea mask?
if (lsmask_type == 1 or lsmask_type == 2): 

    # Read In the Land-Sea Mask 
    print(':: Reading in the Land-Sea Mask.')
    lslat,lslon,lsmask = read_lsmask.main(land_sea) 

    # Ensure that Land-Sea Mask Longitudes are in Range 0-360
    neglon_idx = np.where(lslon<0.)
    lslon[neglon_idx] = lslon[neglon_idx] + 360.
 
    # Determine the Land-Sea Mask (1' Resolution) From ETOPO1 (and Optionally GSHHG as well)
    print(':: Interpolating Land-Sea Mask onto Grid.')
    lsmk = interpolate_lsmask.main(llat,llon,lslat,lslon,lsmask)

    # Apply Land-Sea Mask
    print(':: Applying Land-Sea Mask to the Grids.')
    if (lsmask_type == 1): # mask out land and retain ocean
        llat = llat[lsmk == 0]
        llon = llon[lsmk == 0]
        unit_area = unit_area[lsmk == 0]
        print(':: Total Number of Ocean Elements: %6d' %(len(llat)))
        xtr_str = "_landmask"
    elif (lsmask_type == 2): # mask out ocean and retain land
        llat = llat[lsmk == 1]
        llon = llon[lsmk == 1]
        unit_area = unit_area[lsmk == 1]
        print(':: Total Number of Land Elements: %6d' %(len(llat)))
        xtr_str = "_oceanmask"
else:
    xtr_str = ""

# Output Load Cells to File for Use with LoadDef
if (write_nc == True):
    print(":: Writing netCDF-formatted file.")
    outname = ("commonMesh_global_" + str(gspace_lat) + "_" + str(gspace_lon) + "_" + str(slat_inn) + "_" + str(nlat_inn) + "_" + \
        str(wlon_inn) + "_" + str(elon_inn) + "_" + str(enhanced_lat_inn) + "_" + str(enhanced_lon_inn) + xtr_str + ".nc")
    outfile = ("../../output/Grid_Files/nc/commonMesh/" + outname)
    # Open new NetCDF file in "write" mode
    dataset = netCDF4.Dataset(outfile,'w',format='NETCDF4_CLASSIC')
    # Define dimensions for variables
    numpts = len(llat)
    midpoint_lat = dataset.createDimension('midpoint_lat',numpts)
    midpoint_lon = dataset.createDimension('midpoint_lon',numpts)
    unit_area_patch = dataset.createDimension('unit_area_patch',numpts)
    # Create variables 
    midpoint_lats = dataset.createVariable('midpoint_lat',float,('midpoint_lat',))
    midpoint_lons = dataset.createVariable('midpoint_lon',float,('midpoint_lon',))
    unit_area_patches = dataset.createVariable('unit_area_patch',float,('unit_area_patch',))
    # Add units
    midpoint_lats.units = 'degree_north'
    midpoint_lons.units = 'degree_east'
    unit_area_patches.units = 'dimensionless (need to multiply by r^2 when used)'
    # Assign data
    midpoint_lats[:] = llat
    midpoint_lons[:] = llon
    unit_area_patches[:] = unit_area
    # Write Data to File
    dataset.close()
if (write_txt == True):
    print(":: Writing plain-text file.")
    outname = ("commonMesh_global_" + str(gspace_lat) + "_" + str(gspace_lon) + "_" + str(slat_inn) + "_" + str(nlat_inn) + "_" + \
        str(wlon_inn) + "_" + str(elon_inn) + "_" + str(enhanced_lat_inn) + "_" + str(enhanced_lon_inn) + ".txt")
    outfile = ("../../output/Grid_Files/text/commonMesh/" + outname)
    # Prepare Data
    all_data = np.array(list(zip(llat,llon,unit_area)), dtype=[('llat',float),('llon',float),('unit_area',float)])
    # Write Data to File
    np.savetxt(outfile, all_data, fmt=["%.15f",]*3, delimiter="      ")

# Plot
plt.plot(llon,llat,'.',ms=6)
plt.show()


