# *********************************************************************
# FUNCTION TO INTERPOLATE LAND-SEA MASK ONTO THE TEMPLATE GRID
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

import scipy as sc
import numpy as np
import sys
import time
from scipy import interpolate

def main(ilat,ilon,lslat,lslon,lsmask):

    # Ensure Load Model Longitudes are in Range of 0-360 (To Match the Integration Mesh)
    neglon_idx = np.where(lslon<0.)
    lslon[neglon_idx] = lslon[neglon_idx] + 360.

    # To Avoid Edge Effects at the Greenwich Meridian, Add a Few Columns of Overlap Beyond 360deg and Before 0deg
    degrees_of_overlap = 1.
    near_zero = np.where(lslon<=degrees_of_overlap); near_zero = near_zero[0] # Find indices of longitude values near zero
    near_end = np.where(lslon>=(360.-degrees_of_overlap)); near_end = near_end[0] # Find indices of longitude values near 360deg
    # Copy Values Over to Either Side
    lslat_list = lslat.tolist(); lslon_list = lslon.tolist(); lsmask_list = lsmask.tolist()
    for aa in range(0,len(near_zero)):
        cidx = near_zero[aa]
        lslat_list.append(lslat[cidx]); lslon_list.append(lslon[cidx] + 360.); lsmask_list.append(lsmask[cidx])
    for bb in range(0,len(near_end)):
        cidx = near_end[bb]
        lslat_list.append(lslat[cidx]); lslon_list.append(lslon[cidx] - 360.); lsmask_list.append(lsmask[cidx])
    # Convert Back to Arrays
    lslat = np.asarray(lslat_list); lslon = np.asarray(lslon_list); lsmask = np.asarray(lsmask_list)

    # Interpolate Land-Sea Mask onto Integration Mesh By Nearest-Neighbor Interpolation (Retain 1=Land; 0=Ocean)
    #start = time.clock()
    #print 'Interpolation Using Griddata'
    lsmk = interpolate.griddata((lslat,lslon),lsmask,(ilat,ilon),method='nearest')
    #end = time.clock()
    #print "%.2gs" % (end-start)
 
    # Return Land-Sea Mask at Integration Mesh Points
    return lsmk

