# *********************************************************************
# FUNCTION TO CONVERT TEMPLATE GRID COORDINATES TO GEOGRAPHIC COORDs
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

def main(rlat,rlon,ldel,lazm,unit_area):

    # Determine Geographic Coordinates of Mesh Midpoints
    # e.g. www.movable-type.co.uk/scripts/latlong.html
    # e.g. www.geomidpoint.com/destination/calculation.html
    xv,yv = sc.meshgrid(ldel,lazm)
    ldelfull = xv.flatten()
    lazmfull = yv.flatten()
    rlatrad = rlat*(pi/180.)
    rlonrad = rlon*(pi/180.)
    ldelrad = np.multiply(ldelfull,(pi/180.))
    lazmrad = np.multiply(lazmfull,(pi/180.))
    ilat    = np.arcsin(np.sin(rlatrad)*np.cos(ldelrad) + np.cos(rlatrad)*np.sin(ldelrad)*np.cos(lazmrad))
    ilon    = rlonrad + np.arctan2(np.sin(lazmrad)*np.sin(ldelrad)*np.cos(rlatrad), \
         np.cos(ldelrad) - np.sin(rlatrad)*np.sin(ilat)) 
     
    # Convert Back to Degrees
    ilat = np.multiply(ilat,(180./pi))
    ilon = np.multiply(ilon,(180./pi))

    # Determine Unit Areas for Each Cell
    xv,yv = sc.meshgrid(unit_area,lazm)
    iarea = xv.flatten()

    # Shift to Range 0-360 
    # Test for Negative Values
    if any(ilon<0.):
        flag1 = True
    else:
        flag1 = False
    while (flag1 == True):
        neglon_idx = np.where(ilon<0.)
        ilon[neglon_idx] = ilon[neglon_idx] + 360.
        if any(ilon<0.):
            flag1 = True
        else:
            flag1 = False
    # Test for Greater than 360
    if any(ilon>360.):
        flag2 = True
    else: 
        flag2 = False
    while (flag2 == True):
        lrglon_idx = np.where(ilon>360.)
        ilon[lrglon_idx] = ilon[lrglon_idx] - 360.  
        if any(ilon>360.):
            flag2 = True
        else:
            flag2 = False

    # Return Lat/Lon for Integration Mesh Grid Midpoints
    return ilat,ilon,iarea

