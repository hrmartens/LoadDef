# *********************************************************************
# FUNCTION TO READ IN ETOPO1 DATA FILES
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

from __future__ import print_function
import numpy as np
import scipy as sc
#from scipy.io import netcdf
import netCDF4
import sys
from GRDGEN.utility import make_lsmask_sparse
import matplotlib.pyplot as plt
#import time

# NOTE: REQUIRES NETCDF4 MODULE + LIBCURL.SO.3 

def main(filename,show_figures=False):
    
    # Read In NetCDF File
    f = netCDF4.Dataset(filename)
    #print(f.dimensions['x'])
    #print(f.dimensions['y'])
    #print(f.variables)
    lon = f.variables['x'][:]
    lat = f.variables['y'][:]
    elev = f.variables['z'][:]
    f.close()

    # Mask Out Land Grid-Points
    land_mask = np.ma.masked_greater_equal(elev,0.)

    # Replace Masked Data with Fill Value of 1 (LAND = 1)
    land_mask = np.ma.filled(land_mask,fill_value=1.)

    # Mask Out Ocean Grid-Points
    ocean_mask = np.ma.masked_less(land_mask,0.)
 
    # Replace Masked Data with Fill Value of 0 (OCEAN = 0)
    lsmask = np.ma.filled(ocean_mask,fill_value=0.)

    # Save Arrays in Original Format
    lon1dseq = lon.copy()
    lat1dseq = lat.copy()
    lsmask2darr = lsmask.copy()
    if (show_figures == True):
        plt.imshow(lsmask2darr,origin="lower")
        plt.show()
 
    # Reformat Elevation Points into 1D Vectors
    grid_olon, grid_olat = sc.meshgrid(lon,lat)
    olon = grid_olon.flatten()
    olat = grid_olat.flatten()
    lsmask  = lsmask.flatten()

    # Make the Land-Sea Mask Sparse within xx Grid Points of the Coastline
    xx = 5
    olat,olon,lsmask = make_lsmask_sparse.main(lat,lon,lsmask2darr,xx)

    # Return Parameters
    return olat,olon,lsmask,lat1dseq,lon1dseq,lsmask2darr

