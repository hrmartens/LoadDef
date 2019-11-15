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
from math import pi
import sys
import netCDF4
import time

def main(filename,lf_format,delim=None,regular_grid=False):

    # Read in Data File (plain-text format)
    if (lf_format == "txt"):
        lat = []; lon = []; amp = []; pha = []
        with open(filename) as fln:
            for myl in fln:
                myvars = myl.split(delim)
                lat.append(float(myvars[0])); lon.append(float(myvars[1]))
                amp.append(float(myvars[2])); pha.append(float(myvars[3]))
        lat = np.asarray(lat); lon = np.asarray(lon)
        amp = np.asarray(amp); pha = np.asarray(pha)
    # Read in Data File (netCDF format)
    elif (lf_format == "nc"):
        f = netCDF4.Dataset(filename)
        lon = f.variables['longitude'][:]
        lat = f.variables['latitude'][:]
        amp = f.variables['amplitude'][:]
        pha = f.variables['phase'][:]
        f.close()
    else:
        print(':: Error. Invalid file format for load model. [read_AmpPha.py]')
 
    # Ensure Load Model Longitudes are in Range of 0-360
    neglon_idx = np.where(lon<0.)
    lon[neglon_idx] = lon[neglon_idx] + 360.

    # Write Amplitude to New Array to Make it Writeable
    amp  = np.divide(amp,1.)

    # Write Phase to New Array to Make it Writeable 
    pha  = np.divide(pha,1.)

    if (regular_grid == True):
        # Save Arrays in Original Format
        lon1dseq = np.unique(lon)
        lat1dseq = np.unique(lat)
        amp2darr = np.empty((len(lat1dseq),len(lon1dseq)))
        pha2darr = np.empty((len(lat1dseq),len(lon1dseq)))
        # Determine Indices of Unique 1d Arrays that Correspond to Lat/Lon Values in Original 1d Arrays
        myidxlat = np.searchsorted(lat1dseq,lat)
        myidxlon = np.searchsorted(lon1dseq,lon)
        amp2darr[myidxlat,myidxlon] = amp
        pha2darr[myidxlat,myidxlon] = pha
    else:
        lat1dseq = lon1dseq = amp2darr = pha2darr = None
 
    # Return Parameters
    return lat,lon,amp,pha,lat1dseq,lon1dseq,amp2darr,pha2darr

