# *********************************************************************
# FUNCTION TO READ IN A LOAD MODEL
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
import sys
import netCDF4

def main(filename):

    # Read in Data File 
    f = netCDF4.Dataset(filename)
    lon = f.variables['longitude'][:]
    lat = f.variables['latitude'][:]
    real = f.variables['real'][:]
    imag = f.variables['imag'][:]
    iarea = f.variables['area'][:]
    f.close()
 
    # Ensure Load Model Longitudes are in Range of 0-360
    neglon_idx = np.where(lon<0.)
    lon[neglon_idx] = lon[neglon_idx] + 360.

    # Write Amplitude to New Array to Make it Writeable
    real  = np.divide(real,1.)

    # Write Phase to New Array to Make it Writeable 
    imag  = np.divide(imag,1.)

    # Write Area to New Array to Make it Writeable
    iarea = np.divide(iarea,1.)

    # Return Parameters
    return lat,lon,real,imag,iarea

