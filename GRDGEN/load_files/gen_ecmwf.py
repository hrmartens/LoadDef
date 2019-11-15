#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO GENERATE GRIDS FOR ATMOSPHERIC LOADS FROM ECMWF
# :: GRIDS GENERATED MAY BE USED BY LOADDEF (run_cn.py) OR IN GMT
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

# MODIFY PYTHON PATH TO INCLUDE 'CONVGF' DIRECTORY
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd() + "/../../")

# IMPORT PYTHON MODULES
import numpy as np
import scipy as sc
import datetime
import netCDF4
from GRDGEN.utility import read_ecmwf

# --------------- SPECIFY USER INPUTS --------------------- #

# Atmospheric Surface Pressure Files from ECMWF - MUST HAVE NETCDF4 FOR PYTHON INSTALLED 
#  Specify the directory containing the yearly netcdf files here:
ecmwf_directory = ("../../input/Load_Models/ECMWF/Surface_Pressure/")

# Date Range for Temporal-Mean Computation (yyyy, mm, dd); End Day is Included (Files to be Read in)
start_year_tm = 2016; start_month_tm = 10; start_day_tm = 1
end_year_tm = 2017; end_month_tm = 10; end_day_tm = 1
 
# Date Range for Output Files (yyyy, mm, dd); End Day is Included (Files to be Written out)
start_year_out = 2016; start_month_out = 10; start_day_out = 1
end_year_out = 2017; end_month_out = 10; end_day_out = 1

# Remove spatial and temporal averages?
rm_spatial_mean = False
rm_temporal_mean = False

# Order in which to remove the temporal (t) and spatial (s) averages (false = t then s; true = s then t)
flip = False

# Additional Name Tag
tmrange = "%4d%02d%02d-%4d%02d%02d" % (start_year_tm, start_month_tm, start_day_tm, end_year_tm, end_month_tm, end_day_tm)
add_tag = (tmrange + "_ECMWF")

# Write Load Information to a netCDF-formatted File? (Default for convolution)
write_nc = True

# Write Load Information to a Text File? (Alternative for convolution)
write_txt = False

# Write Load Information to a GMT-formatted File? (Lon, Lat, Amplitude)
write_gmt = False

# ------------------ END USER INPUTS ----------------------- #

# -------------------- BEGIN CODE -------------------------- #

# Check for output of a file
if (write_nc == False) and (write_txt == False) and (write_gmt == False):
    print(":: Error: No output file(s) selected. Options: netCDF, GMT, and/or plain-text.")
    sys.exit()

# Create Folders
if not (os.path.isdir("../../output/Grid_Files/")):
    os.makedirs("../../output/Grid_Files/")
if not (os.path.isdir("../../output/Grid_Files/GMT/")):
    os.makedirs("../../output/Grid_Files/GMT/")
if not (os.path.isdir("../../output/Grid_Files/GMT/ATML/")):
    os.makedirs("../../output/Grid_Files/GMT/ATML/")
if not (os.path.isdir("../../output/Grid_Files/nc/")):
    os.makedirs("../../output/Grid_Files/nc/")
if not (os.path.isdir("../../output/Grid_Files/nc/ATML/")):
    os.makedirs("../../output/Grid_Files/nc/ATML/")
if not (os.path.isdir("../../output/Grid_Files/text/")):
    os.makedirs("../../output/Grid_Files/text/")
if not (os.path.isdir("../../output/Grid_Files/text/ATML/")):
    os.makedirs("../../output/Grid_Files/text/ATML/")

# Filename Tags
if (flip == False): # temporal then spatial
    tag = ("rmTM1" + str(rm_temporal_mean) + "_rmSM2" + str(rm_spatial_mean) + "_" + add_tag + "_")
else: # spatial then temporal
    tag = ("rmSM1" + str(rm_spatial_mean) + "_rmTM2" + str(rm_temporal_mean) + "_" + add_tag + "_")

# Determine Ordinal Dates for Temporal Mean Calculation
mydate1 = datetime.datetime(start_year_tm, start_month_tm, start_day_tm,00,00,00) #start_date = datetime.date.toordinal(mydate1)
mydate2 = datetime.datetime(end_year_tm, end_month_tm, end_day_tm,00,00,00) #end_date = datetime.date.toordinal(mydate2)
# Determine Date Range (From Start to End, Increasing by 6 Hours)
delta = datetime.timedelta(hours=6)
curr = mydate1
date_list = []
date_list.append(curr)
while curr < mydate2:
    curr += delta
    date_list.append(curr)

# Determine Ordinal Dates for Output
mydate1 = datetime.datetime(start_year_out, start_month_out, start_day_out,00,00,00)
mydate2 = datetime.datetime(end_year_out, end_month_out, end_day_out,00,00,00)
# Determine Date Range (From Start to End, Increasing by 6 Hours)
delta = datetime.timedelta(hours=6)
curr = mydate1
date_list_out = []
date_list_out.append(curr)
while curr < mydate2:
    curr += delta
    date_list_out.append(curr)

# Determine Number of Dates for Temporal Mean
if isinstance(date_list,float) == True:
    numel = 1
else: 
    numel = len(date_list)

# Determine Number of Dates for Output
if isinstance(date_list_out,float) == True:
    numel_out = 1
else:
    numel_out = len(date_list_out)

# Check Number of Dates
if (numel_out > numel):
    print(':: Warning: Fewer Dates for the Temporal Mean Computation than for the Output Files.')
elif (min(date_list) > min(date_list_out)):
    print(':: Warning: Dates for Output Files are Outside the Range of the Files to be Read in.')
elif (max(date_list) < max(date_list_out)):
    print(':: Warning: Dates for Output Files are Outside the Range of the Files to be Read in.')

# Create Array of String Dates
string_dates = []
for qq in range(0,numel):
    mydt = date_list[qq]
    string_dates.append(mydt.strftime('%Y%m%d%H%M%S'))

# Fill Amplitude Array
to_mask = np.empty((480*241,len(date_list)))
atml_amp = np.empty((480*241,len(date_list)))
dates_to_delete = []
# SHAPE OF ARRAY
atml_shape = atml_amp.shape
# Loop Through Dates
for ii in range(0,len(date_list)):
    mydt = date_list[ii] 
    string_date = mydt.strftime('%Y%m%d%H%M%S') # Convert Date to String in YYYY-mm-dd-HH-MM-SS Format
    print(':: Reading %s' %(string_date))
    string_year = mydt.strftime('%Y') # Convert Date to String in YYYY Format
    string_month = mydt.strftime('%m') # Month
    # Complete Pathname to Current ECMWF File
    loadfile = ecmwf_directory + "ECMWF-ERA-Interim-" + string_year + "-" + string_month + ".nc"
    # Read the File 
    if (os.path.isfile(loadfile)):
        llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_ecmwf.main(loadfile,mydt)
        # Combine Amplitude and Phase 
        to_mask[:,ii] = np.zeros((atml_shape[0],)) # file exists : do not apply mask
        atml_amp[:,ii] = np.multiply(amp,np.cos(np.radians(pha)))
    else: # File Does Not Exist
        print(':: Warning: File Does Not Exist.')
        # Save date_list index, then continue
        dates_to_delete.append(ii)
        continue

# Delete dates with no data
date_list = np.delete(date_list,dates_to_delete)
string_dates = np.delete(string_dates,dates_to_delete)
atml_amp = np.delete(atml_amp,dates_to_delete,axis=1)
to_mask = np.delete(to_mask,dates_to_delete,axis=1)

# Order of Removing the Temporal and Spatial Means
if (flip == False): # Temporal then Spatial
 
    # COMPUTE TEMPORAL MEAN
    if (rm_temporal_mean == True):
        atml_temporal_avg = np.average(atml_amp,axis=1)
        print(':: Computing temporal mean.')
        # Put Averages into Array | Efficient, but Memory Problems for lots of Dates ...
        #atml_temporal_avg_array = np.tile(atml_temporal_avg,(atml_shape[1],1)).T
        # Subtract Temporal Array
        for jj in range(0,len(date_list)): # Loop Takes Longer, but Saves on Memory
            #atml_amp[:,jj] = np.subtract(atml_amp[:,jj], atml_temporal_avg_array[:,jj])
            atml_amp[:,jj] = np.subtract(atml_amp[:,jj], atml_temporal_avg)
    atml_temporal_avg_array = atml_temporal_avg = None

    # COMPUTE SPATIAL MEAN
    if (rm_spatial_mean == True):
        # Mask the Array when Computing Spatial Averages
        masked_amp = np.ma.masked_where(to_mask == 1,atml_amp)
        atml_spatial_avg = np.ma.average(masked_amp,axis=0)
        # Convert Back to Numpy Array
        atml_spatial_avg = np.ma.filled(atml_spatial_avg,fill_value=0.)
        print(':: Computing spatial mean.')
        # Put Averages into Array | Efficient, but Memory Problems for lots of Dates ...
        #atml_spatial_avg_array = np.tile(atml_spatial_avg,(atml_shape[0],1))
        # Subtract Spatial Array
        for kk in range(0,len(date_list)): # Loop Takes Longer, but Saves on Memory
            #atml_amp[:,kk] = np.subtract(atml_amp[:,kk], atml_spatial_avg_array[:,kk])
            atml_amp[:,kk] = np.subtract(atml_amp[:,kk], atml_spatial_avg[kk])
    atml_spatial_avg_array = atml_spatial_avg = None

else: # Spatial then Temporal

    # COMPUTE SPATIAL MEAN
    if (rm_spatial_mean == True):
        # Mask the Array when Computing Spatial Averages
        masked_amp = np.ma.masked_where(to_mask == 1,atml_amp)
        atml_spatial_avg = np.ma.average(masked_amp,axis=0)
        # Convert Back to Numpy Array
        atml_spatial_avg = np.ma.filled(atml_spatial_avg,fill_value=0.)
        print(':: Computing spatial mean.')
        # Put Averages into Array | Efficient, but Memory Problems for lots of Dates ...
        #atml_spatial_avg_array = np.tile(atml_spatial_avg,(atml_shape[0],1))
        # Subtract Spatial Array
        for kk in range(0,len(date_list)): # Loop Takes Longer, but Saves on Memory
            #atml_amp[:,kk] = np.subtract(atml_amp[:,kk], atml_spatial_avg_array[:,kk])
            atml_amp[:,kk] = np.subtract(atml_amp[:,kk], atml_spatial_avg[kk])
    atml_spatial_avg_array = atml_spatial_avg = None

    # COMPUTE TEMPORAL MEAN
    if (rm_temporal_mean == True):
        atml_temporal_avg = np.average(atml_amp,axis=1)
        print(':: Computing temporal mean.')
        # Put Averages into Array
        #atml_temporal_avg_array = np.tile(atml_temporal_avg,(atml_shape[1],1)).T
        # Subtract Temporal Array | Efficient, but Memory Problems for lots of Dates ...
        for jj in range(0,len(date_list)): # Loop Takes Longer, but Saves on Memory
            #atml_amp[:,jj] = np.subtract(atml_amp[:,jj], atml_temporal_avg_array[:,jj])
            atml_amp[:,jj] = np.subtract(atml_amp[:,jj], atml_temporal_avg)
    atml_temporal_avg_array = atml_temporal_avg = None

# Convert to Masked Array (Re-set all masked grid points to zero)
masked_amp = np.ma.masked_where(to_mask == 1,atml_amp)
print(':: Masking the amplitude array.')
for bb in range(0,len(date_list)): # Loop Takes Longer, but Saves on Memory
    atml_amp[:,bb] = np.ma.filled(masked_amp[:,bb],fill_value=0.)
masked_amp = to_mask = None

# Set Phase to Zero (Amplitudes Contain Phase)
atml_pha = np.zeros((480*241,len(date_list)))

# Loop Through Dates and Write to File
for kk in range(0,len(date_list_out)):
    
    # Output ATML Grid to File for Plotting in GMT
    mydt = date_list_out[kk]
    # Convert Date to String in YYYY-mm-dd-HH-MM-SS Format
    string_date = mydt.strftime('%Y%m%d%H%M%S')
    # Find Consistent Dates
    if not string_date in string_dates:
        continue
    # Locate Date in Full Date List
    jj = np.where(string_date == np.asarray(string_dates)); jj = jj[0][0]
    idx = str(jj) # Convert to string to test if a value exists (including zero)
    if not idx:
        print(':: Warning: No Date Match Found for Output Date in Range of Amplitude Array | %s' %(string_date))
        continue
    jj = int(idx) # Convert back to integer to use as index
    # Prepare to Write to File
    print(':: Writing %s' %(string_dates[jj]))
    atml_out = ("atml_" + tag + string_date + ".txt")
    atml_out_nc = ("atml_" + tag + string_date + ".nc")
    atml_file = ("../../output/Grid_Files/GMT/ATML/height-anomaly_" + atml_out)
    atml_file_pressure = ("../../output/Grid_Files/GMT/ATML/pressure_" + atml_out)
    atml_file_nc = ("../../output/Grid_Files/nc/ATML/convgf_" + atml_out_nc)
    atml_file_text = ("../../output/Grid_Files/text/ATML/convgf_" + atml_out)
    # Prepare Data
    all_atml_data = np.column_stack((llon,llat,atml_amp[:,jj]))
    all_atml_data_pressure = np.column_stack((llon,llat,atml_amp[:,jj]*9.81))
    all_atml_data_convgf = np.column_stack((llat,llon,atml_amp[:,jj],atml_pha[:,jj]))
    # Write Files
    if (write_nc == True):
        print(":: Writing netCDF-formatted file.")
        # Open new NetCDF file in "write" mode
        dataset = netCDF4.Dataset(atml_file_nc,'w',format='NETCDF4_CLASSIC')
        # Define dimensions for variables
        num_pts = len(llat)
        latitude = dataset.createDimension('latitude',num_pts)
        longitude = dataset.createDimension('longitude',num_pts)
        amplitude = dataset.createDimension('amplitude',num_pts)
        phase = dataset.createDimension('phase',num_pts)
        # Create variables
        latitudes = dataset.createVariable('latitude',np.float,('latitude',))
        longitudes = dataset.createVariable('longitude',np.float,('longitude',))
        amplitudes = dataset.createVariable('amplitude',np.float,('amplitude',))
        phases = dataset.createVariable('phase',np.float,('phase',))
        # Add units
        latitudes.units = 'degree_north'
        longitudes.units = 'degree_east'
        amplitudes.units = 'm'
        phases.units = 'degree'
        # Assign data
        latitudes[:] = llat
        longitudes[:] = llon
        amplitudes[:] = atml_amp[:,jj]
        phases[:] = atml_pha[:,jj]
        # Write Data to File
        dataset.close()
    if (write_gmt == True):
        print(":: Writing to GMT-convenient text file.")
        np.savetxt(atml_file, all_atml_data, fmt='%f %f %f')
        np.savetxt(atml_file_pressure, all_atml_data_pressure, fmt='%f %f %f')
    if (write_txt == True):
        print(":: Writing to plain-text file.")
        np.savetxt(atml_file_text, all_atml_data_convgf, fmt='%f %f %f %f')

if (write_gmt == True):

    print(":: Writing to GMT-convenient text file.")
    # Compute Standard Deviation for Each Amplitude Pixel
    atml_std = np.std(atml_amp,axis=1)
    # Export Standard Deviation to File
    atml_out = "atml_std_" + string_dates[0] + "_" + string_dates[-1] + ".txt"
    atml_file = ("../../output/Grid_Files/GMT/ATML/height-anomaly_" + atml_out)
    atml_file_pressure = ("../../output/Grid_Files/GMT/ATML/pressure_" + atml_out)
    # Prepare Data
    all_atml_data = np.column_stack((llon,llat,atml_std))
    all_atml_data_pressure = np.column_stack((llon,llat,atml_std*9.81))
    # Write Data to File
    np.savetxt(atml_file, all_atml_data, fmt='%f %f %f')
    np.savetxt(atml_file_pressure, all_atml_data_pressure, fmt='%f %f %f')

    # Compute Maximum for Each Amplitude Pixel
    atml_abs = np.absolute(atml_amp)
    atml_max = np.amax(atml_abs,axis=1)
    # Export Standard Deviation to File
    atml_out = "atml_max_" + string_dates[0] + "_" + string_dates[-1] + ".txt"
    atml_file = ("../../output/Grid_Files/GMT/ATML/height-anomaly_" + atml_out)
    atml_file_pressure = ("../../output/Grid_Files/GMT/ATML/pressure_" + atml_out)
    # Prepare Data
    all_atml_data = np.column_stack((llon,llat,atml_max))
    all_atml_data_pressure = np.column_stack((llon,llat,atml_max*9.81))
    # Write Data to File
    np.savetxt(atml_file, all_atml_data, fmt='%f %f %f')
    np.savetxt(atml_file_pressure, all_atml_data_pressure, fmt='%f %f %f')

# --------------------- END CODE --------------------------- #


