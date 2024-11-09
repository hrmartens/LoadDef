# *********************************************************************
# FUNCTION TO GENERATE THE POLE-CENTERED TEMPLATE GRID
#  DESIGNED FOR TWO SYMMETRIC DISK LOADS (ONE AT EACH POLE)
# 
# Copyright (c) 2020-2024: HILARY R. MARTENS
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
from math import pi
import netCDF4 
from CONVGF.utility import read_lsmask
from CONVGF.CN import interpolate_lsmask
import matplotlib.pyplot as plt
from CONVGF.CN import intmesh2geogcoords

# --------------- SPECIFY USER INPUTS --------------------- #

# -- Mesh Paramters --
del1 = 0.001    # increment in angular resolution (degrees) for zone 1
del2 = 0.005    # increment in angular resolution for zone 2
del3 = 0.01     # increment in angular resolution for zone 3
del4 = 0.1      # increment in angular resolution for zone 4
del5 = 0.5      # increment in angular resolution for zone 5
del6 = 0.1      # increment in angular resolution for zone 6
del7 = 0.01     # increment in angular resolution for zone 7
del8 = 0.005    # increment in angular resolution for zone 8
del9 = 0.001    # increment in angular resolution for zone 9
z1 = 11.0       # outer edge of zone 1 (degrees)
z2 = 15.0       # outer edge of zone 2
z3 = 20.0       # outer edge of zone 3
z4 = 30.0       # outer edge of zone 4
z5 = 150.0      # outer edge of zone 5
z6 = 160.0      # outer edge of zone 6
z7 = 165.0      # outer edge of zone 7
z8 = 169.0      # outer edge of zone 8
azm = 0.5       # increment in azimuthal resolution (degrees)
 
# 7. Land-Sea Mask
#  :: 0 = do not mask ocean or land (retain full model); 1 = mask out land (retain ocean); 2 = mask out oceans (retain land)
#  :: Recommended: 1 for oceanic; 2 for atmospheric
lsmask_type = 0
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

def create_grid(delinc1=0.001,delinc2=0.005,delinc3=0.01,delinc4=0.1,delinc5=0.5,delinc6=0.1,delinc7=0.01,delinc8=0.005,delinc9=0.001,\
         z1b=11.0,z2b=15.0,z3b=20.0,z4b=30.0,z5b=150.0,z6b=160.0,z7b=165.0,z8b=169.0,azinc=0.5):

    """

    # -- DEFINITIONS -- 
    # delinc1 = angular distance increment for zone 1
    # delinc2 = angular distance increment for zone 2
    # delinc3 = angular distance increment for zone 3
    # delinc4 = angular distance increment for zone 4
    # delinc5 = angular distance increment for zone 5
    # delinc6 = angular distance increment for zone 6
    # delinc7 = angular distance increment for zone 7
    # delinc8 = angular distance increment for zone 8
    # azinc   = azimuthal increment
    # z1b     = zone 1 boundary (degrees)
    # z2b     = zone 2 boundary
    # z3b     = zone 3 boundary
    # z4b     = zone 4 boundary
    # z5b     = zone 5 boundary
    # z6b     = zone 6 boundary
    # z7b     = zone 7 boundary
    # z8b     = zone 8 boundary

    # -- Default Mesh Paramters --
    delinc1 : angular distance (degrees) increment for zone 1
        Default is 0.001

    delinc2 : angular distance (degrees) increment for zone 2
        Default is 0.005

    delinc3 : angluar distance (degrees) increment for zone 3
        Default is 0.01

    delinc4 : angluar distance (degrees) increment for zone 4
        Default is 0.1

    delinc5 : angluar distance (degrees) increment for zone 5
        Default is 0.5

    delinc6 : angular distance (degrees) increment for zone 6
        Default is 0.1

    delinc7 : angular distance (degrees) increment for zone 7
        Default is 0.01

    delinc8 : angular distance (degrees) increment for zone 8
        Default is 0.005

    delinc9 : angular distance (degrees) increment for zone 9
        Default is 0.001

    z1b     : zone 1 boundary (degrees; 0<z1b<z2b)
        Default is 11.0

    z2b     : zone 2 boundary (degrees; z1b<z2b<z3b)
        Default is 15.0

    z3b     : zone 3 boundary (degrees; z2b<z3b<z4b)
        Default is 20.0

    z4b     : zone 4 boundary (degrees; z3b<z4b<z5b)
        Default is 30.0

    z5b     : zone 5 boundary (degrees; z4b<z5b<180)
        Default is 150.0

    z6b     : zone 6 boundary (degrees; z5b<z6b<z7b)
        Default is 160.0

    z7b     : zone 7 boundary (degrees; z6b<z7b<z8b)
        Default is 165.0

    z8b     : zone 8 boundary (degrees; z7b<z8b<180)
        Default is 169.0

    azinc  : azimuthal increment # NOTE: Maybe should match azminc with delinc5 (i.e., keep azminc consistent with theta increment at 90 degrees from station,
                                #       where azimuth and theta increments are consistent in horizontal distance along planet's surface)
                                #       :: azminc*sin(theta) ~ delinc
        Default is 0.5 

    """

    # Determine Cell Grid Lines
    inum1 = int((z1b/delinc1)+1.)                      # number of increments, zone 1
    ldel1 = np.linspace(0.,z1b,num=inum1)              # delta values for zone 1
    inum2 = int((z2b-z1b)/delinc2)
    ldel2 = np.linspace(z1b+delinc2,z2b,num=inum2)
    inum3 = int((z3b-z2b)/delinc3)
    ldel3 = np.linspace(z2b+delinc3,z3b,num=inum3)
    inum4 = int((z4b-z3b)/delinc4)
    ldel4 = np.linspace(z3b+delinc4,z4b,num=inum4)
    inum5 = int((z5b-z4b)/delinc5)
    ldel5 = np.linspace(z4b+delinc5,z5b,num=inum5)
    inum6 = int((z6b-z5b)/delinc6)                     
    ldel6 = np.linspace(z5b+delinc6,z6b,num=inum6)     
    inum7 = int((z7b-z6b)/delinc7)                     
    ldel7 = np.linspace(z6b+delinc7,z7b,num=inum7)     
    inum8 = int((z8b-z7b)/delinc8)                     
    ldel8 = np.linspace(z7b+delinc8,z8b,num=inum8)     
    inum9 = int((180.0-z8b)/delinc9)                     # number of increments, final zone
    ldel9 = np.linspace(z8b+delinc9,180.0,num=inum9)     # delta values for final zone
    gldel = np.concatenate([ldel1,ldel2,ldel3,ldel4,ldel5,ldel6,ldel7,ldel8,ldel9]) # delta values for inner and outer zones
    inuma = int((360./azinc)+1.)                        # number of azimuthal increments
    lazm  = np.linspace(0.,360.,num=inuma)              # azimuthal values for mesh
    glazm = lazm[0:-1]                                  # don't include last element (same as first element)    

    # Determine Unit Area of Each Cell
    unit_area = []
    all_del = np.concatenate([ldel1,ldel2,ldel3,ldel4,ldel5,ldel6,ldel7,ldel8,ldel9])
    all_del_rad = np.multiply(all_del,(pi/180.))
    azinc_rad = np.multiply(azinc,(pi/180.))
    for ii in range(1,len(all_del_rad)):
        unit_area.append(np.multiply(azinc_rad,\
            np.cos(all_del_rad[ii-1])-np.cos(all_del_rad[ii])))
    unit_area = np.asarray(unit_area)

    # Determine Cell Midpoints
    azm_mdpts = lazm + azinc/2.                        # midpoints between azimuthal gridlines
    azm_mdpts = azm_mdpts[0:-1]                        # don't include value > 360.
    del_mdpts1 = ldel1 + delinc1/2.                    # midpoints between delta gridlines, inner zone
    del_mdpts1 = del_mdpts1[0:-1]                      # don't include value > izb
    del_mdpts2 = ldel2 + delinc2/2.                    # midpoints between delta gridlines, zone-2
    del_mdpts2 = del_mdpts2[0:-1]                      # don't include value > z2b    
    del_mdpts3 = ldel3 + delinc3/2.
    del_mdpts3 = del_mdpts3[0:-1]
    del_mdpts4 = ldel4 + delinc4/2.
    del_mdpts4 = del_mdpts4[0:-1]
    del_mdpts5 = ldel5 + delinc5/2.
    del_mdpts5 = del_mdpts5[0:-1]
    del_mdpts6 = ldel6 + delinc6/2.
    del_mdpts6 = del_mdpts6[0:-1]
    del_mdpts7 = ldel7 + delinc7/2.
    del_mdpts7 = del_mdpts7[0:-1]
    del_mdpts8 = ldel8 + delinc8/2.
    del_mdpts8 = del_mdpts8[0:-1]
    del_mdpts9 = ldel9 + delinc9/2.
    del_mdpts9 = del_mdpts9[0:-1]
    del_mdpt_1to2 = z1b + delinc2/2.              # midpoint for cell between zone-1 and zone-2 refinement sections
    del_mdpts1 = np.append(del_mdpts1,del_mdpt_1to2)
    del_mdpt_2to3 = z2b + delinc3/2.
    del_mdpts2 = np.append(del_mdpts2,del_mdpt_2to3)
    del_mdpt_3to4 = z3b + delinc4/2.
    del_mdpts3 = np.append(del_mdpts3,del_mdpt_3to4)
    del_mdpt_4to5 = z4b + delinc5/2.
    del_mdpts4 = np.append(del_mdpts4,del_mdpt_4to5)
    del_mdpt_5to6 = z5b + delinc6/2.
    del_mdpts5 = np.append(del_mdpts5,del_mdpt_5to6)
    del_mdpt_6to7 = z6b + delinc7/2.
    del_mdpts6 = np.append(del_mdpts6,del_mdpt_6to7)
    del_mdpt_7to8 = z7b + delinc8/2.
    del_mdpts7 = np.append(del_mdpts7,del_mdpt_7to8)
    del_mdpt_8to9 = z8b + delinc9/2.
    del_mdpts8 = np.append(del_mdpts8,del_mdpt_8to9)
    ldel = np.concatenate([del_mdpts1,del_mdpts2,del_mdpts3,del_mdpts4,del_mdpts5,del_mdpts6,del_mdpts7,del_mdpts8,del_mdpts9])
    lazm = azm_mdpts 

    # Create the grid
    xv,yv = np.meshgrid(ldel,lazm)
    lcolat = xv.flatten()
    llon = yv.flatten()

    # Convert co-latitude to latitude
    llat = 90.0 - lcolat

    # Determine Unit Areas for Each Cell
    xv,yv = np.meshgrid(unit_area,lazm)
    unit_area = xv.flatten()

    # Return Delta/Azm Values for Gridlines and Midpoints
    return llat,llon,ldel,lazm,unit_area

# Create the grid
llat,llon,ldel,lazm,unit_area = create_grid(delinc1=del1,delinc2=del2,delinc3=del3,delinc4=del4,delinc5=del5,delinc6=del6,delinc7=del7,delinc8=del8,delinc9=del9,\
         z1b=z1,z2b=z2,z3b=z3,z4b=z4,z5b=z5,z6b=z6,z7b=z7,z8b=z8,azinc=0.5)

# Compute Geographic Coordinates of Integration Mesh Cell Midpoints
#llat,llon,unit_area = intmesh2geogcoords.main(89.9999999,0.0000001,ldel,lazm,unit_area)
#print(llat)
#print(llon)
#print(unit_area)

# Plot
#plt.plot(llon,llat,'.',ms=6)
#plt.show()

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
if write_nc:
    print(":: Writing netCDF-formatted file.")
    outname = ("commonMesh_polarDisks_" + str(del1) + "-" + str(z1) + "_" + \
               str(del2) + "-" + str(z2) + "_" + \
               str(del3) + "-" + str(z3) + "_" + \
               str(del4) + "-" + str(z4) + "_" + \
               str(del5) + "-" + str(z5) + "_" + \
               str(del6) + "-" + str(z6) + "_" + \
               str(del7) + "-" + str(z7) + "_" + \
               str(del8) + "-" + str(z8) + "_" + \
               str(del9) + xtr_str + ".nc")
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
if write_txt:
    print(":: Writing plain-text file.")
    outname = ("commonMesh_polarDisks_" + str(del1) + "-" + str(z1) + "_" + \
               str(del2) + "-" + str(z2) + "_" + \
               str(del3) + "-" + str(z3) + "_" + \
               str(del4) + "-" + str(z4) + "_" + \
               str(del5) + "-" + str(z5) + "_" + \
               str(del6) + "-" + str(z6) + "_" + \
               str(del7) + "-" + str(z7) + "_" + \
               str(del8) + "-" + str(z8) + "_" + \
               str(del9) + "_" + xtr_str + ".txt")
    outfile = ("../../output/Grid_Files/text/commonMesh/" + outname)
    # Prepare Data
    all_data = np.array(list(zip(llat,llon,unit_area)), dtype=[('llat',float),('llon',float),('unit_area',float)])
    # Write Data to File
    np.savetxt(outfile, all_data, fmt=["%.15f",]*3, delimiter="      ")

# Print file name
print(':: New mesh file: ', outfile)

# Plot
plt.plot(llon,llat,'.',ms=6)
plt.show()
