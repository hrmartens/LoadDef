# *********************************************************************
# FUNCTION TO READ IN THE FES2004 OCEAN TIDE MODELS
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

import numpy as np
import scipy as sc
from scipy.io import netcdf

def main(filename):
    
    # Read In NetCDF File
    f = netcdf.netcdf_file(filename, 'r')
    # print f.dimensions
    # print f.variables
    lon = f.variables['longitude']
    lat = f.variables['latitude']
    amp = f.variables['amplitude']
    pha = f.variables['phase']
    f.close()

    # Access Arrays
    lon = lon[:]
    lat = lat[:]
    amp = amp[:]
    pha = pha[:]

    # Convert Amplitude from Centimeters to Meters
    amp  = np.divide(amp,100.)

    # Write Phase to New Array to Make it Writeable 
    pha  = np.divide(pha,1.) 

    # NaN 'FillValue' for FES2004 is 1.84e+19 (in cm)
    #  Set NaN Values to Zero
    amp[amp > 10.**10.] = 0.
    pha[pha > 10.**10.] = 0.

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

    # NaN 'FillValue' for FES2004 is 1.84e+19 (in cm)
    #  Set NaN Values to Zero
    nanidx = np.where(amp > 10**10)
    amp[nanidx] = 0.
    pha[nanidx] = 0.

    # Return Parameters
    return olat,olon,amp,pha,lat1dseq,lon1dseq,amp2darr,pha2darr

