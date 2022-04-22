#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO GENERATE A CUSTOM LOAD GRID
# :: GRIDS GENERATED MAY BE USED BY LOADDEF (run_cn.py) OR IN GMT
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
import netCDF4
from CONVGF.utility import read_AmpPha

# --------------- SPECIFY USER INPUTS --------------------- #
 
# Specify Bounding Box for Load Model (e.g. boundingbox.klokantech.com)
#  :: In general, the longitude range should be [0,360]
#  :: In the special case that the bounding box crosses the prime meridian,
#     the range should be [-180,0] for wlon and [0,180] for elon
wlon=40. # range [0,360] | If Bounding Box Crosses Prime Meridian, range = [-180,0]
elon=80. # range [0,360] | If Bounding Box Crosses Prime Meridian, range = [0,180]
slat=-40.  # range [-90,90]
nlat=10.  # range [-90,90]

# Apply Prime-Meridian Correction?
#  :: Set to "True" if the Bounding Box Stradles the Prime Meridian
#  :: Ranges Must be [-180,0] and [0,180]
pm_correct = False

# Specify Load Height (meters)
loadamp=1.0

# Specify Phase (deg)
loadpha=0.0

# Apply the Given Load Height and Phase to the Region Outside the Bounding Box
set_outside = True

# Apply the Given Load Height and Phase to the Region Inside the Bounding Box
set_inside = False

# Optional: Starting Grid File (If no initial starting grid, set "initial_grid = None")
initial_grid = None
regular_grid = True

# Grid Spacing
#  :: Only used if a starting grid file is not supplied
gspace = 0.25

# Output Filename
outfile = ("custom")

# Write Load Information to a netCDF-formatted File? (Default for convolution)
write_nc = True

# Write Load Information to a Text File? (Alternative for convolution)
write_txt = False

# Write Load Information to a GMT-formatted File? (Lon, Lat, Amplitude)
write_gmt = False

# ------------------ END USER INPUTS ----------------------- #

# -------------------- BEGIN CODE -------------------------- #

# Check for output of a file
if (write_nc == False) and (write_txt == False) and (write_gmt == False):
    print(":: Error: No output file(s) selected. Options: netCDF, GMT, and/or plain-text.")
    sys.exit()

# Create Folders
if not (os.path.isdir("../../output/Grid_Files/")):
    os.makedirs("../../output/Grid_Files/")
if not (os.path.isdir("../../output/Grid_Files/GMT/")):
    os.makedirs("../../output/Grid_Files/GMT/")
if not (os.path.isdir("../../output/Grid_Files/GMT/Custom/")):
    os.makedirs("../../output/Grid_Files/GMT/Custom/")
if not (os.path.isdir("../../output/Grid_Files/nc/")):
    os.makedirs("../../output/Grid_Files/nc/")
if not (os.path.isdir("../../output/Grid_Files/nc/Custom/")):
    os.makedirs("../../output/Grid_Files/nc/Custom/")
if not (os.path.isdir("../../output/Grid_Files/text/")):
    os.makedirs("../../output/Grid_Files/text/")
if not (os.path.isdir("../../output/Grid_Files/text/Custom/")):
    os.makedirs("../../output/Grid_Files/text/Custom/")

# Read in Starting Grid, or Generate a New Grid
if initial_grid is not None:
    # Read in Starting Grid
    llat,llon,amp,pha,lat1dseq,lon1dseq,amp2darr,pha2darr = read_AmpPha.main(initial_grid,regular_grid=regular_grid)
# Generate New Grid
else:
    lats = np.arange(-90.,90.,gspace) + (gspace/2.0)
    lons = np.arange(0.,360.,gspace) + (gspace/2.0)
    xv,yv = np.meshgrid(lons,lats)
    llon = np.ravel(xv)
    llat = np.ravel(yv)
    amp = np.zeros((len(llon),))
    pha = np.zeros((len(llon),))

# If Necessary, Apply Prime-Meridian Correction (Shift to Range [-180,180])
if (pm_correct == True):
    print(':: Applying the prime-meridian correction.')
    if wlon > 0.:
        sys.exit('Error: When applying the prime-meridian correction, the longitudes of the bounding box must range from [-180,180].')
    if elon < 0.:
        sys.exit('Error: When applying the prime-meridian correction, the longitudes of the bounding box must range from [-180,180].')
    pm_correction = np.where(llon>=180.); pm_correction = pm_correction[0]
    llon[pm_correction] -= 360.

# Find Indices Outside Bouding Box
bboxoutside = np.where((llon <= wlon) | (llat <= slat) | (llon >= elon) | (llat >= nlat)); bboxoutside = bboxoutside[0]

# Find Indices Inside Bounding Box
bboxinside = np.where((llon >= wlon) & (llat >= slat) & (llon <= elon) & (llat <= nlat)); bboxinside = bboxinside[0]

# If Necessary, Shift Longitude Values back to Original Range ([0,360])
if (pm_correct == True):
    llon[pm_correction] += 360.

# Set Amplitude and Phase to Constant Values Outside Bounding Box
if (set_outside == True):
    amp[bboxoutside] = loadamp
    pha[bboxoutside] = loadpha

# Set Amplitude and Phase to Constant Values Inside Bounding Box
if (set_inside == True):
    amp[bboxinside] = loadamp
    pha[bboxinside] = loadpha

# Output OTL Grid to File for Plotting in GMT
if (write_gmt == True):
    print(":: Writing GMT-convenient text file.")
    custom_out = outfile + ".txt"
    custom_file = ("../../output/Grid_Files/GMT/Custom/height-anomaly_" + custom_out)
    # Prepare Data
    all_custom_data = np.column_stack((llon,llat,amp))
    # Write Data to File
    np.savetxt(custom_file, all_custom_data, fmt='%f %f %f')

# Output Grid to File for Use with LoadDef
if (write_nc == True):
    print(":: Writing netCDF-formatted file.")
    custom_out = (outfile + ".nc")
    custom_file = ("../../output/Grid_Files/nc/Custom/convgf_" + custom_out)
    # Open new NetCDF file in "write" mode
    dataset = netCDF4.Dataset(custom_file,'w',format='NETCDF4_CLASSIC')
    # Define dimensions for variables
    num_pts = len(llat)
    latitude = dataset.createDimension('latitude',num_pts)
    longitude = dataset.createDimension('longitude',num_pts)
    amplitude = dataset.createDimension('amplitude',num_pts)
    phase = dataset.createDimension('phase',num_pts)
    # Create variables
    latitudes = dataset.createVariable('latitude',float,('latitude',))
    longitudes = dataset.createVariable('longitude',float,('longitude',))
    amplitudes = dataset.createVariable('amplitude',float,('amplitude',))
    phases = dataset.createVariable('phase',float,('phase',))
    # Add units
    latitudes.units = 'degree_north'
    longitudes.units = 'degree_east'
    amplitudes.units = 'm'
    phases.units = 'degree'
    # Assign data
    latitudes[:] = llat
    longitudes[:] = llon
    amplitudes[:] = amp
    phases[:] = pha
    # Write Data to File
    dataset.close()
if (write_txt == True):
    print(":: Writing plain-text file.")
    custom_out = (outfile + ".txt")
    custom_file = ("../../output/Grid_Files/text/Custom/convgf_" + custom_out)
    # Prepare Data
    all_custom_data = np.column_stack((llat,llon,amp,pha))
    # Write Data to File
    np.savetxt(custom_file, all_custom_data, fmt='%f %f %f %f')

# --------------------- END CODE --------------------------- #


