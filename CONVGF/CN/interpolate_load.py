# *********************************************************************
# FUNCTION TO INTERPOLATE THE LOAD MODEL ONTO THE TEMPLATE GRID
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
from scipy import interpolate
# basemap package has now been replaced by "cartopy"
# from mpl_toolkits.basemap import interp # If no basemap module, comment out and use gridddata below instead
from math import pi
import sys

def main(ilat,ilon,llat,llon,lreal,limag,regular):

    # Ensure Load Model Longitudes are in Range of 0-360 (To Match the Integration Mesh)
    neglon_idx = np.where(llon<0.)
    llon[neglon_idx] = llon[neglon_idx] + 360.

    # To Avoid Edge Effects at the Greenwich Meridian, Add a Few Columns of Overlap Beyond 360deg and Before 0deg
    degrees_of_overlap = 1.
    near_zero = np.where((llon<=degrees_of_overlap) & (llon>0.)); near_zero = near_zero[0] # Find indices of longitude values near zero
    near_end = np.where((llon>=(360.-degrees_of_overlap)) & (llon<360.)); near_end = near_end[0] # Find indices of longitude values near 360deg
    # Copy Values Over to Either Side
    llat_list = llat.tolist(); llon_list = llon.tolist(); lreal_list = lreal.tolist(); limag_list = limag.tolist()
    for aa in range(0,len(near_zero)):
        cidx = near_zero[aa]
        llat_list.append(llat[cidx]); llon_list.append(llon[cidx] + 360.)
        lreal_list.append(lreal[cidx]); limag_list.append(limag[cidx])
    for bb in range(0,len(near_end)):
        cidx = near_end[bb] 
        llat_list.append(llat[cidx]); llon_list.append(llon[cidx] - 360.)
        lreal_list.append(lreal[cidx]); limag_list.append(limag[cidx]) 
    # Convert Back to Arrays
    llat = np.asarray(llat_list); llon = np.asarray(llon_list)
    lreal = np.asarray(lreal_list); limag = np.asarray(limag_list)

    # Interpolate
    if (regular == True):
        # Convert to Grid
        llon1dseq = np.unique(llon)
        llat1dseq = np.unique(llat)
        real2darr = np.empty((len(llat1dseq),len(llon1dseq)))
        imag2darr = np.empty((len(llat1dseq),len(llon1dseq)))
        test = ((len(llon1dseq) * len(llat1dseq)) - (len(lreal)))
        if (test != 0):
            print(':: WARNING: Grid might be irregular, but user specified regular. May yield incorrect results.')
        # Determine Indices of Unique 1d Arrays that Correspond to Lat/Lon Values in Original 1d Arrays
        myidxllat = np.searchsorted(llat1dseq,llat)
        myidxllon = np.searchsorted(llon1dseq,llon)
        real2darr[myidxllat,myidxllon] = lreal
        imag2darr[myidxllat,myidxllon] = limag
        # OPTION 1: Basemap
        # Interpolate Using Basemap
        #ic1 = interp(real2darr,llon1dseq,llat1dseq,ilon,ilat,order=1)
        #ic2 = interp(imag2darr,llon1dseq,llat1dseq,ilon,ilat,order=1)
        # OPTION 2: RectBivariateSpline (Slower than Basemap, but only requires SciPy)
        # Build Interpolator
        flreal = interpolate.RectBivariateSpline(llat1dseq,llon1dseq,real2darr,kx=1,ky=1,s=0) 
        flimag = interpolate.RectBivariateSpline(llat1dseq,llon1dseq,imag2darr,kx=1,ky=1,s=0) 
        # Evaluate Interpolator
        ic1 = flreal.ev(ilat,ilon)
        ic2 = flimag.ev(ilat,ilon)
    else:
        # OPTION 1: Interpolate Using Griddata (Faster than LinearNDInterpolator)
        ic1 = interpolate.griddata((llat,llon),lreal,(ilat,ilon),method='linear',fill_value=0.)
        ic2 = interpolate.griddata((llat,llon),limag,(ilat,ilon),method='linear',fill_value=0.)
        # OPTION 2: Build Interpolator
        #flreal = interpolate.LinearNDInterpolator((llat,llon),lreal,fill_value=0.)
        #flimag = interpolate.LinearNDInterpolator((llat,llon),limag,fill_value=0.)
        # Evaluate Interpolator
        #ic1 = flreal(ilat,ilon)
        #ic2 = flimag(ilat,ilon)

    # Return Loads at Integration Mesh Points
    return ic1,ic2




# Alternative Interpolation Methods:
# 1. Basemap Interp
#   from mpl_toolkits.basemap import interp
#    ic1 = interp(real2darr,llon1dseq,llat1dseq,ilon,ilat,order=1)
#    ic2 = interp(imag2darr,llon1dseq,llat1dseq,ilon,ilat,order=1)
# 2. RectBivariateSpline (Slower than Basemap, but Uses only Scipy)
#    # Interpolate Using RectBivariateSpline
#    flreal = sc.interpolate.RectBivariateSpline(llat1dseq,llon1dseq,real2darr,kx=1,ky=1,s=0) # K=1: LINEAR; K=3: CUBIC SPLINE
#    flimag = sc.interpolate.RectBivariateSpline(llat1dseq,llon1dseq,imag2darr,kx=1,ky=1,s=0) # K=1: LINEAR; K=3: CUBIC SPLINE
#    # Acquire Values at Integration Mesh Midpoints
#    ic1 = flreal.ev(ilat,ilon)
#    ic2 = flimag.ev(ilat,ilon)
# 3. scipy.ndimage.interpolation.map_coordinates
#    x0 = min(llon1dseq); y0 = min(llat1dseq)
#    dx = llon1dseq[1]-llon1dseq[0]; dy = llat1dseq[1]-llat1dseq[0]
#    xvals = (ilon-x0)/dx; yvals = (ilat-y0)/dy
#    coords = sc.array([yvals,xvals])
#    ic1 = ndimage.map_coordinates(real2darr,coords,order=1)
#    ic2 = ndimage.map_coordinates(imag2darr,coords,order=1)
# 4. scipy.interpolate.LinearNDInterpolator (Slower than scipy.interpolate.griddata)       
#    # Build Interpolator
#    flreal = interpolate.LinearNDInterpolator((llat,llon),lreal)
#    flimag = interpolate.LinearNDInterpolator((llat,llon),limag)
#    # Evaluate Interpolator
#    ic1 = flreal(ilat,ilon)
#    ic2 = flimag(ilat,ilon)

