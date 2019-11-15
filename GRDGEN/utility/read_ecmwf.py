# *********************************************************************
# FUNCTION TO READ ECMWF SURFACE PRESSURE NETCDF FILES
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
import sys
import datetime

# NOTE: REQUIRES NETCDF4 MODULE + LIBCURL.SO.3 

# e.g.: filename = ("../sp-20070101-20071231.nc")
 
def main(filename,current_date):
    
    # Read In NetCDF File
    f = netCDF4.Dataset(filename)
    #print(f.dimensions['longitude'])
    #print(f.dimensions['latitude'])
    #print(f.variables)
    lon = f.variables['longitude'][:]
    lat = f.variables['latitude'][:]
    time = f.variables['time'][:]
    sp = f.variables['sp'][:]
    pha = np.zeros((len(lat),len(lon)))
    sp_info = f.variables['sp']
    time_info = f.variables['time']
    #print(sp_info)
    f.close()  

    # Obtained from: sp_info = f.variables['sp']; print(sp_info)
    #  Set Fill Values to Zero
    sp[sp == -32767] = 0.

    # Convert Surface Pressure to Density*Meters by Dividing by 'g'
    sp  = np.divide(sp,9.81)

    # Convert Times (in Hours Since 1900-01-01 00:00:00.0) to Python Datetimes
    original_date = datetime.datetime(1900,1,1,0,0,0)
    my_dates = []
    diff_dates = []
    diff_dates_sec = []
    for ii in range(0,len(time)):
        my_dates.append(original_date + datetime.timedelta(hours=float(time[ii])))
        diff_dates.append(my_dates[ii]-current_date)
        diff_dates_sec.append(datetime.timedelta.total_seconds(diff_dates[ii]))
    diff_dates_sec = np.asarray(diff_dates_sec)

    # Select Desired Date 
    dselect = np.where(diff_dates_sec == 0.0); dselect = dselect[0]; dselect = dselect[0]

    # Save Arrays in Original Format
    lon1dseq = lon
    lat1dseq = lat
    sp3darr  = sp
    amp2darr = sp3darr[dselect,:,:]
    pha2darr = pha

    # Arrange Latitude from South to North
    lat1dseq = np.flipud(lat1dseq)
    amp2darr = np.flipud(amp2darr)
    pha2darr = np.flipud(pha2darr)

    # Reformat Load Points into 1D Vectors
    grid_olon, grid_olat = sc.meshgrid(lon1dseq,lat1dseq)
    olon = grid_olon.flatten()
    olat = grid_olat.flatten()
    amp  = amp2darr.flatten()
    pha  = pha2darr.flatten()

    # Return Parameters
    return olat,olon,amp,pha,lat1dseq,lon1dseq,amp2darr,pha2darr


