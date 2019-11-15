# *********************************************************************
# FUNCTION TO READ IN THE LAND-SEA MASK
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
import sys

def main(filename,delim=None):

    # Read in Data File
    lat = []; lon = []; msk = []
    with open(filename) as fln:
        for myl in fln:
            myvars = myl.split(delim)
            lat.append(float(myvars[0])); lon.append(float(myvars[1]))
            msk.append(float(myvars[2]))
    lat = np.asarray(lat); lon = np.asarray(lon)
    msk = np.asarray(msk)

    # Ensure Load Model Longitudes are in Range of 0-360
    neglon_idx = np.where(lon<0.)
    lon[neglon_idx] = lon[neglon_idx] + 360.

    # Return Parameters
    return lat,lon,msk

