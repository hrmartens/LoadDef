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
# basemap package has now been replaced by "cartopy" (original basemap.interp function copied below)
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
        # Interpolate Using Basemap (see function copied below)
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


# Basemap is now deprecated. 
def interp(datain,xin,yin,xout,yout,checkbounds=False,masked=False,order=1):
    """
    Interpolate data (``datain``) on a rectilinear grid (with x = ``xin``
    y = ``yin``) to a grid with x = ``xout``, y= ``yout``.

    .. tabularcolumns:: |l|L|

    ==============   ====================================================
    Arguments        Description
    ==============   ====================================================
    datain           a rank-2 array with 1st dimension corresponding to
                     y, 2nd dimension x.
    xin, yin         rank-1 arrays containing x and y of
                     datain grid in increasing order.
    xout, yout       rank-2 arrays containing x and y of desired output grid.
    ==============   ====================================================

    .. tabularcolumns:: |l|L|

    ==============   ====================================================
    Keywords         Description
    ==============   ====================================================
    checkbounds      If True, values of xout and yout are checked to see
                     that they lie within the range specified by xin
                     and xin.
                     If False, and xout,yout are outside xin,yin,
                     interpolated values will be clipped to values on
                     boundary of input grid (xin,yin)
                     Default is False.
    masked           If True, points outside the range of xin and yin
                     are masked (in a masked array).
                     If masked is set to a number, then
                     points outside the range of xin and yin will be
                     set to that number. Default False.
    order            0 for nearest-neighbor interpolation, 1 for
                     bilinear interpolation, 3 for cublic spline
                     (default 1). order=3 requires scipy.ndimage.
    ==============   ====================================================

    .. note::
     If datain is a masked array and order=1 (bilinear interpolation) is
     used, elements of dataout will be masked if any of the four surrounding
     points in datain are masked.  To avoid this, do the interpolation in two
     passes, first with order=1 (producing dataout1), then with order=0
     (producing dataout2).  Then replace all the masked values in dataout1
     with the corresponding elements in dataout2 (using numpy.where).
     This effectively uses nearest neighbor interpolation if any of the
     four surrounding points in datain are masked, and bilinear interpolation
     otherwise.

    Returns ``dataout``, the interpolated data on the grid ``xout, yout``.
    """
    # xin and yin must be monotonically increasing.
    if xin[-1]-xin[0] < 0 or yin[-1]-yin[0] < 0:
        raise ValueError('xin and yin must be increasing!')
    if xout.shape != yout.shape:
        raise ValueError('xout and yout must have same shape!')
    # check that xout,yout are
    # within region defined by xin,yin.
    if checkbounds:
        if xout.min() < xin.min() or \
           xout.max() > xin.max() or \
           yout.min() < yin.min() or \
           yout.max() > yin.max():
            raise ValueError('yout or xout outside range of yin or xin')
    # compute grid coordinates of output grid.
    delx = xin[1:]-xin[0:-1]
    dely = yin[1:]-yin[0:-1]
    if max(delx)-min(delx) < 1.e-4 and max(dely)-min(dely) < 1.e-4:
        # regular input grid.
        xcoords = (len(xin)-1)*(xout-xin[0])/(xin[-1]-xin[0])
        ycoords = (len(yin)-1)*(yout-yin[0])/(yin[-1]-yin[0])
    else:
        # irregular (but still rectilinear) input grid.
        xoutflat = xout.flatten(); youtflat = yout.flatten()
        ix = (np.searchsorted(xin,xoutflat)-1).tolist()
        iy = (np.searchsorted(yin,youtflat)-1).tolist()
        xoutflat = xoutflat.tolist(); xin = xin.tolist()
        youtflat = youtflat.tolist(); yin = yin.tolist()
        xcoords = []; ycoords = []
        for n,i in enumerate(ix):
            if i < 0:
                xcoords.append(-1) # outside of range on xin (lower end)
            elif i >= len(xin)-1:
                xcoords.append(len(xin)) # outside range on upper end.
            else:
                xcoords.append(float(i)+(xoutflat[n]-xin[i])/(xin[i+1]-xin[i]))
        for m,j in enumerate(iy):
            if j < 0:
                ycoords.append(-1) # outside of range of yin (on lower end)
            elif j >= len(yin)-1:
                ycoords.append(len(yin)) # outside range on upper end
            else:
                ycoords.append(float(j)+(youtflat[m]-yin[j])/(yin[j+1]-yin[j]))
        xcoords = np.reshape(xcoords,xout.shape)
        ycoords = np.reshape(ycoords,yout.shape)
    # data outside range xin,yin will be clipped to
    # values on boundary.
    if masked:
        xmask = np.logical_or(np.less(xcoords,0),np.greater(xcoords,len(xin)-1))
        ymask = np.logical_or(np.less(ycoords,0),np.greater(ycoords,len(yin)-1))
        xymask = np.logical_or(xmask,ymask)
    xcoords = np.clip(xcoords,0,len(xin)-1)
    ycoords = np.clip(ycoords,0,len(yin)-1)
    # interpolate to output grid using bilinear interpolation.
    if order == 1:
        xi = xcoords.astype(np.int32)
        yi = ycoords.astype(np.int32)
        xip1 = xi+1
        yip1 = yi+1
        xip1 = np.clip(xip1,0,len(xin)-1)
        yip1 = np.clip(yip1,0,len(yin)-1)
        delx = xcoords-xi.astype(np.float32)
        dely = ycoords-yi.astype(np.float32)
        dataout = (1.-delx)*(1.-dely)*datain[yi,xi] + \
                  delx*dely*datain[yip1,xip1] + \
                  (1.-delx)*dely*datain[yip1,xi] + \
                  delx*(1.-dely)*datain[yi,xip1]
    elif order == 0:
        xcoordsi = np.around(xcoords).astype(np.int32)
        ycoordsi = np.around(ycoords).astype(np.int32)
        dataout = datain[ycoordsi,xcoordsi]
    elif order == 3:
        try:
            from scipy.ndimage import map_coordinates
        except ImportError:
            raise ValueError('scipy.ndimage must be installed if order=3')
        coords = [ycoords,xcoords]
        dataout = map_coordinates(datain,coords,order=3,mode='nearest')
    else:
        raise ValueError('order keyword must be 0, 1 or 3')
    if masked:
        newmask = ma.mask_or(ma.getmask(dataout), xymask)
        dataout = ma.masked_array(dataout, mask=newmask)
        if not isinstance(masked, bool):
            dataout = dataout.filled(masked)
    return dataout


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

