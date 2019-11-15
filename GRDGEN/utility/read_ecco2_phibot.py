# *********************************************************************
# FUNCTION TO READ IN ECCO2 PHIBOT NETCDF FILES 
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
from scipy.io import netcdf
import netCDF4
import sys
import matplotlib.pyplot as plt

# NOTE: REQUIRES NETCDF4 MODULE + LIBCURL.SO.3 

def main(filename):
    
    # Read In NetCDF File
    f = netCDF4.Dataset(filename)
    lon = f.variables['LONGITUDE_T'][:]
    lat = f.variables['LATITUDE_T'][:]
    phibot = f.variables['PHIBOT'][:]
    time = f.variables['TIME'][:] # Days since 1992-01-01 00:00:00
    pha = np.zeros((len(lat),len(lon)))
    f.close()
    #plt.imshow(phibot[0,:,:])
    #plt.show()
    #print(len(phibot[0,:,0])) # number of rows (latitude cells)
    #print(len(phibot[0,0,:])) # number of columns (longitude cells)
    #print(len(phibot[1,0,:])) # index 1 is out of bounds

    # Compute Spatial Average and Remove
#    phibot_flatten = phibot[0,:,:].flatten() # Flatten into a 1D array
#    phibot_spatial_mean = np.mean(phibot_flatten) # Compute mean of 1D array
#    phibot = np.subtract(phibot,phibot_spatial_mean) # Subtract mean value from 2d array

    # Get Mask
    cmask = np.ma.getmask(phibot[0,:,:])

    # Replace Masked Data with Fill Values
    phibot = np.ma.filled(phibot[0,:,:],fill_value=0.)
    #plt.imshow(phibot)
    #plt.show()

    # Convert PHIBOT (m^2/s^2) to AMPLITUDE (m) -- Divide by g
    amp = np.divide(phibot, 9.81)

    # Save Arrays in Original Format
    lon1dseq = lon
    lat1dseq = lat
    amp2darr = amp
    pha2darr = pha

    # Reformat Load Points into 1D Vectors
    grid_olon, grid_olat = sc.meshgrid(lon,lat)
    olon = grid_olon.flatten()
    olat = grid_olat.flatten()
    amp  = amp.flatten()
    pha  = pha.flatten()

    # Return Parameters
    return olat,olon,amp,pha,lat1dseq,lon1dseq,amp2darr,pha2darr,cmask

