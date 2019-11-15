# *********************************************************************
# FUNCTION TO READ IN GRACE MASS GRIDS
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
 
#def main(filename,current_date):
def main(filename,current_date,ldfl2=None,ldfl3=None,scl=None,avsolns=False,appscaling=False): 
   
    # Read In NetCDF File
    f = netCDF4.Dataset(filename)
    #print(f.variables)
    #print(f.dimensions['lon'])
    #print(f.dimensions['lat'])
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    time = f.variables['time'][:]
    lwe = f.variables['lwe_thickness'][:]
    pha = np.zeros((len(lat),len(lon)))
    lat_info = f.variables['lat']
    lwe_info = f.variables['lwe_thickness'] # centimeters
    time_info = f.variables['time']
    #print(lwe_info)
    f.close()  

    # Optionally Read in Additional Solutions
    if ldfl2 is not None:
        # Read NetCDF file
        f2 = netCDF4.Dataset(ldfl2)
        lon2 = f2.variables['lon'][:]
        lat2 = f2.variables['lat'][:]
        time2 = f2.variables['time'][:]
        lwe2 = f2.variables['lwe_thickness'][:]
        pha2 = np.zeros((len(lat2),len(lon2)))
        lwe2_info = f2.variables['lwe_thickness']
        time2_info = f2.variables['time']
        #print(lwe2_info)
        #print(time2_info)
        f2.close()
    if ldfl3 is not None:
        f3 = netCDF4.Dataset(ldfl3)
        lon3 = f3.variables['lon'][:]
        lat3 = f3.variables['lat'][:]
        time3 = f3.variables['time'][:]
        lwe3 = f3.variables['lwe_thickness'][:]
        pha3 = np.zeros((len(lat3),len(lon3)))
        lwe3_info = f3.variables['lwe_thickness']
        time3_info = f3.variables['time']
        #print(time3_info)
        f3.close()

    # Optionally Read in Scaling Factors
    if scl is not None:
        fscl = netCDF4.Dataset(scl)
        #print(fscl.variables)
        lonscl = fscl.variables['Longitude'][:]
        latscl = fscl.variables['Latitude'][:]
        scale_factor = fscl.variables['SCALE_FACTOR'][:]
        msmt_err = fscl.variables['MEASUREMENT_ERROR'][:]
        lkg_err = fscl.variables['LEAKAGE_ERROR'][:]
        fscl.close()

    # Obtained from: lwe_info = f.variables['lwe']; print(lwe_info)
    #  Set Fill Values to Zero
    lwe = np.ma.filled(lwe,fill_value=0.)
    # Convert LWE Thickness from cm to meters
    lwe = np.divide(lwe,100.)
  
    if ldfl2 is not None:
        # Obtained from: lwe_info = f.variables['lwe']; print(lwe_info)
        #  Set Fill Values to Zero
        lwe2 = np.ma.filled(lwe2,fill_value=0.)
        # Convert LWE Thickness from cm to meters
        lwe2 = np.divide(lwe2,100.)
    if ldfl3 is not None:
        # Obtained from: lwe_info = f.variables['lwe']; print(lwe_info)
        #  Set Fill Values to Zero
        lwe3 = np.ma.filled(lwe3,fill_value=0.)
        # Convert LWE Thickness from cm to meters
        lwe3 = np.divide(lwe3,100.) 

    # Convert Times (in Days Since 2002-01-01 00:00:00.0) to Python Datetimes
    original_date = datetime.datetime(2002,1,1,0,0,0)
    my_dates = []
    year_month_day = []
    diff_dates = []
    diff_dates_sec = []
    for ii in range(0,len(time)):
        my_dates.append(original_date + datetime.timedelta(days=float(time[ii])))
        # Find Grace file that matches year, month, and day
        #year_month.append(datetime.datetime(my_dates[ii].year,my_dates[ii].month,01,00,00,00))
        year_month_day.append(datetime.datetime(my_dates[ii].year,my_dates[ii].month,my_dates[ii].day,0,0,0))
        diff_dates.append(year_month_day[ii]-current_date)
        diff_dates_sec.append(datetime.timedelta.total_seconds(diff_dates[ii]))
    diff_dates_sec = np.asarray(diff_dates_sec)

    # Select Desired Date 
    dselect = np.where(diff_dates_sec == 0.0); dselect = dselect[0]
    if (len(dselect > 0)):
        dselect = dselect[0]
    else:
        return None, None, None, None, None, None, None, None

    # Save Arrays in Original Format
    lon1dseq = lon.copy()
    lat1dseq = lat.copy()
    lwe3darr = lwe.copy()
    amp2darr = lwe3darr[dselect,:,:]
    pha2darr = pha.copy()

    # Optionally Compute Average Solutions
    if (avsolns == True):
        if ldfl2 is not None:
            if (len(time) == len(time2)):
                amp2darr2 = lwe2[dselect,:,:]
                if (len(lon) == len(lon2)):
                    if (len(lat) == len(lat2)):
                        amp2darr = (amp2darr + amp2darr2) / 2.
                    else:
                        print('Error: Latitude arrays do not match in length. Cannot average amplitude arrays.')
                else:
                    print('Error: Longitude arrays do not match in length. Cannot average amplitude arrays.')
            else:
                print('Error: Time arrays do not match in length. Cannot extract appropriate values from second solution.')
        if ldfl3 is not None:
            if (len(time) == len(time3)):
                amp2darr3 = lwe3[dselect,:,:]
                if (len(lon) == len(lon3)):
                    if (len(lat) == len(lat3)):
                        amp2darr = (amp2darr + amp2darr3) / 2.
                    else:
                        print('Error: Latitude arrays do not match in length. Cannot average amplitude arrays.')
                else:
                    print('Error: Longitude arrays do not match in length. Cannot average amplitude arrays.')
            else:
                print('Error: Time arrays do not match in length. Cannot extract appropriate values from third solution.')

    # Optionally Apply the Scaling Factors
    if (appscaling == True):
        if scl is not None:
            #  Set Fill Values to Zero
            nztuple = scale_factor.nonzero()
            nz = np.transpose(nztuple)
            entries = nz.shape[0]
            for kk in range(0,entries):
                cval = nz[kk]
                cvalx = cval[0]
                cvaly = cval[1]
                amp2darr[cvalx,cvaly] *= scale_factor[cvalx,cvaly]

    # Reformat Load Points into 1D Vectors
    grid_olon, grid_olat = sc.meshgrid(lon1dseq,lat1dseq)
    olon = grid_olon.flatten()
    olat = grid_olat.flatten()
    amp  = amp2darr.flatten()
    pha  = pha2darr.flatten()

    # Return Parameters
    return olat,olon,amp,pha,lat1dseq,lon1dseq,amp2darr,pha2darr


