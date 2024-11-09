#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO GENERATE BOUNDING BOXES of LOAD CELLS FOR AN INVERSION
# :: GRIDS GENERATED MAY BE USED IN LOADINV
# 
# Copyright (c) 2021-2024: HILARY R. MARTENS
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
import datetime
import netCDF4 
from CONVGF.utility import read_lsmask
from CONVGF.CN import interpolate_lsmask

# --------------- SPECIFY USER INPUTS --------------------- #
 
# 1. Specify the region of interest
slat = 28.
nlat = 50.
wlon = 233. 
elon = 258. 

# 2. Specify the cell size (in degrees)
cell_size = 0.25 
 
# 3. Apply a land-sea mask? 
apply_ls = True
land_sea = ("../../input/Land_Sea/ETOPO1_Ice_g_gmt4_wADD.txt")
ls_mask = 'land' # 'ocean' = keep ocean (including cells with partial ocean); 'land' = keep land (including cells with partial land)
 
# 4. Write Load Information to a netCDF-formatted File? (Default for convolution)
write_nc = True

# 5. Write Load Information to a Text File? (Alternative for convolution)
write_txt = False

# ------------------ END USER INPUTS ----------------------- #

# -------------------- BEGIN CODE -------------------------- #

# Check for output of a file
if (write_nc == False) and (write_txt == False) and (write_gmt == False):
    print(":: Error: No output file(s) selected. Options: netCDF, GMT, and/or plain-text.")
    sys.exit()

# Create Folders
if not (os.path.isdir("../../output/Grid_Files/")):
    os.makedirs("../../output/Grid_Files/")
if not (os.path.isdir("../../output/Grid_Files/nc/")):
    os.makedirs("../../output/Grid_Files/nc/")
if not (os.path.isdir("../../output/Grid_Files/nc/cells/")):
    os.makedirs("../../output/Grid_Files/nc/cells/")
if not (os.path.isdir("../../output/Grid_Files/text/")):
    os.makedirs("../../output/Grid_Files/text/")
if not (os.path.isdir("../../output/Grid_Files/text/cells/")):
    os.makedirs("../../output/Grid_Files/text/cells/")

# Adjust longitudes, if necessary
if (wlon < 0.):
    wlon += 360.
if (elon < 0.):
    elon += 360.

# Create bounding boxes for individual cells
lats = np.arange(slat,(nlat+cell_size),cell_size)
lons = np.arange(wlon,(elon+cell_size),cell_size)
counter = 0
cids = []
slats = []
nlats = []
wlons = []
elons = []
for ii in range (0,len(lats)-1):
    cslat = lats[ii]
    cnlat = lats[ii+1]
    for jj in range(0,len(lons)-1):
        cwlon = lons[jj]
        celon = lons[jj+1]
        cids.append(str(counter).zfill(10))
        slats.append(cslat)
        nlats.append(cnlat)
        wlons.append(cwlon)
        elons.append(celon)
        counter += 1
cids = np.asarray(cids)
slats = np.asarray(slats)
nlats = np.asarray(nlats)
wlons = np.asarray(wlons)
elons = np.asarray(elons)

# Land-Sea Mask
if (apply_ls == True):
 
    # Read In the Land-Sea Mask
    print(':: Reading in the Land-Sea Mask.')
    lslat,lslon,lsmask = read_lsmask.main(land_sea)

    # Ensure that Land-Sea Mask Longitudes are in Range 0-360
    neglon_idx = np.where(lslon<0.)
    lslon[neglon_idx] = lslon[neglon_idx] + 360.

    # Determine the Land-Sea Mask (1' Resolution) From ETOPO1 (and Optionally GSHHG as well)
    print(':: Interpolating Land-Sea Mask onto Grid. Checking all four corners of each grid cell.')
    print(':: Lower-left corner.')
    lsmka = interpolate_lsmask.main(slats,wlons,lslat,lslon,lsmask) 
    print(':: Lower-right corner.')
    lsmkb = interpolate_lsmask.main(slats,elons,lslat,lslon,lsmask) 
    print(':: Upper-left corner.')
    lsmkc = interpolate_lsmask.main(nlats,wlons,lslat,lslon,lsmask)
    print(':: Upper-right corner.')
    lsmkd = interpolate_lsmask.main(nlats,elons,lslat,lslon,lsmask) \
 
    # Apply Land-Sea Mask
    print(':: Applying Land-Sea Mask to the Grid.')
    if (ls_mask == 'land'):
        discard_ids = np.where((lsmka == 0) & (lsmkb == 0) & (lsmkc == 0) & (lsmkd == 0)); discard_ids = discard_ids[0]
    elif (ls_mask == 'ocean'):
        discard_ids = np.where((lsmka == 1) & (lsmkb == 1) & (lsmkc == 1) & (lsmkd == 1)); discard_ids = discard_ids[0]
    else: 
        sys.exit(':: Error. Incorrect land-sea mask code.')
    slats = np.delete(slats,discard_ids)
    nlats = np.delete(nlats,discard_ids)
    wlons = np.delete(wlons,discard_ids)
    elons = np.delete(elons,discard_ids)
    cids = np.delete(cids,discard_ids)
    print(':: Total Number of Cells After Applying Land-Sea Mask: %10d' %(len(slats)))

# Output Load Cells to File for Use with LoadDef
if (write_nc == True):
    print(":: Writing netCDF-formatted file.")
    outname = ("cells_" + str(slat) + "_" + str(nlat) + "_" + str(wlon) + "_" + str(elon) + "_" + str(cell_size) + ".nc")
    outfile = ("../../output/Grid_Files/nc/cells/" + outname)
    # Open new NetCDF file in "write" mode
    dataset = netCDF4.Dataset(outfile,'w',format='NETCDF4_CLASSIC')
    # Define dimensions for variables
    num_pts = len(slats)
    slatitude = dataset.createDimension('slatitude',num_pts)
    nlatitude = dataset.createDimension('nlatitude',num_pts)
    wlongitude = dataset.createDimension('wlongitude',num_pts)
    elongitude = dataset.createDimension('elongitude',num_pts)
    nchars = dataset.createDimension('nchars',10)
    # Create variables
    slatitudes = dataset.createVariable('slatitude',float,('slatitude',))
    nlatitudes = dataset.createVariable('nlatitude',float,('nlatitude',))
    wlongitudes = dataset.createVariable('wlongitude',float,('wlongitude',))
    elongitudes = dataset.createVariable('elongitude',float,('elongitude',))
    cell_ids = dataset.createVariable('cell_ids','S1',('slatitude','nchars'))
    # Add units
    slatitudes.units = 'degree_north'
    nlatitudes.units = 'degree_north'
    wlongitudes.units = 'degree_east'
    elongitudes.units = 'degree_east'
    cell_ids.units = 'string'
    # Assign data
    slatitudes[:] = slats
    nlatitudes[:] = nlats
    wlongitudes[:] = wlons
    elongitudes[:] = elons
    cell_ids._Encoding = 'ascii'
    cell_ids[:] = np.array(cids,dtype='S10')
    # Write Data to File
    dataset.close()
if (write_txt == True):
    print(":: Writing plain-text file.")
    outname = ("cells_" + str(slat) + "_" + str(nlat) + "_" + str(wlon) + "_" + str(elon) + "_" + str(cell_size) + ".txt")
    outfile = ("../../output/Grid_Files/text/cells/" + outname)
    # Prepare Data
    all_data = np.array(list(zip(slats,nlats,wlons,elons,cids)), dtype=[('slats',float),('nlats',float),('wlons',float),('elons',float),('cids','U10')])
    # Write Data to File
    np.savetxt(outfile, all_data, fmt=["%.8f",]*4 + ["%s"], delimiter="      ")
 
# --------------------- END CODE --------------------------- #


