# *********************************************************************
# FUNCTION TO READ IN THE OSU LOCAL OCEAN TIDE MODELS
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
from math import pi
import sys

def main(filename,locdelim):

    # Read in Data File
    lon,lat,amp,pha = np.loadtxt(filename,unpack=True,delimiter=locdelim)

    # Ensure Load Model Longitudes are in Range of 0-360
    neglon_idx = np.where(lon<0.)
    lon[neglon_idx] = lon[neglon_idx] + 360.
    
    # Write Amplitude to New Array to Make it Writeable
    amp  = np.divide(amp,1.)
    
    # Write Phase to New Array to Make it Writeable 
    pha  = np.divide(pha,1.) 
    
    # Save Arrays in Original Format
    lon1dseq = np.unique(lon)
    lat1dseq = np.unique(lat)
    amp2darr = np.empty((len(lat1dseq),len(lon1dseq)))
    pha2darr = np.empty((len(lat1dseq),len(lon1dseq)))
    for ii in range(0,len(amp)):
        mylat = np.where(lat1dseq == lat[ii]); mylat = mylat[0]
        mylon = np.where(lon1dseq == lon[ii]); mylon = mylon[0]
        amp2darr[mylat,mylon] = amp[ii]
        pha2darr[mylat,mylon] = pha[ii]
 
    # Return Parameters
    return lat,lon,amp,pha,lat1dseq,lon1dseq,amp2darr,pha2darr

