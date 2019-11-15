# *********************************************************************
# FUNCTION TO READ IN THE TPXO8-ATLAS OCEAN TIDE MODELS
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
from math import pi
import sys

# NOTE: REQUIRES NETCDF4 MODULE + LIBCURL.SO.3 

def main(filename):
    
    # Read In NetCDF File
    f = netCDF4.Dataset(filename)
    #print(f.variables)
    lon = f.variables['lon_z'][:]
    lat = f.variables['lat_z'][:]
    real = f.variables['hRe'][:]
    imag = f.variables['hIm'][:]
    #real_var = f.variables['hRe']; print(real_var)
    amp = np.absolute(real + np.multiply(imag,1.j))
    pha = np.multiply(np.arctan2(-imag,real),180./pi)
    amp = np.transpose(amp) # transpose the arrays
    pha = np.transpose(pha)
    f.close()
    
    # Replace Masked Data with Fill Values
    amp = np.ma.filled(amp,fill_value=0.)
    pha = np.ma.filled(pha,fill_value=0.)

    # Convert Amplitude from Millimeters to Meters
    amp  = np.divide(amp,1000.)

    # Write Phase to New Array to Make it Writeable 
    pha  = np.divide(pha,1.) 

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
    return olat,olon,amp,pha,lat1dseq,lon1dseq,amp2darr,pha2darr

