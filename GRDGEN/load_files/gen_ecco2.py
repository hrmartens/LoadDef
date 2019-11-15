#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO GENERATE GRIDS FOR NON-TIDAL OCEANIC LOADS FROM ECCO2
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
from GRDGEN.utility import read_ecco2_phibot
import matplotlib.pyplot as plt

# --------------- SPECIFY USER INPUTS --------------------- #

# Bottom Pressure Files from NASA ECCO2 - MUST HAVE NETCDF4 FOR PYTHON INSTALLED 
#  Specify the directory containing the daily netcdf files here:
phibot_directory = ("../../input/Load_Models/ECCO2/PHIBOT/")

# Date Range for Temporal-Mean Computation (yyyy, mm, dd); End Day is Included (To be read in)
# Memory Constraints Seem to Limit the Range to About Four Years
start_year_tm = 2015; start_month_tm = 1; start_day_tm = 1
end_year_tm = 2015; end_month_tm = 3; end_day_tm = 31

# Date Range for Output Files (yyyy, mm, dd); End Day is Included (To be written out)
start_year_out = 2015; start_month_out = 1; start_day_out = 1
end_year_out = 2015; end_month_out = 3; end_day_out = 31

# Remove spatial and temporal averages?
rm_spatial_mean = True
rm_temporal_mean = True

# Order in which to remove the temporal (t) and spatial (s) averages (false = t then s; true = s then t)
flip = False

# Additional Name Tag
tmrange = "%4d%02d%02d-%4d%02d%02d" % (start_year_tm, start_month_tm, start_day_tm, end_year_tm, end_month_tm, end_day_tm)
add_tag = (tmrange + "_ECCO2")

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
if not (os.path.isdir("../../output/Grid_Files/GMT/NTOL/")):
    os.makedirs("../../output/Grid_Files/GMT/NTOL/")
if not (os.path.isdir("../../output/Grid_Files/nc/")):
    os.makedirs("../../output/Grid_Files/nc/")
if not (os.path.isdir("../../output/Grid_Files/nc/NTOL/")):
    os.makedirs("../../output/Grid_Files/nc/NTOL/")
if not (os.path.isdir("../../output/Grid_Files/text/")):
    os.makedirs("../../output/Grid_Files/text/")
if not (os.path.isdir("../../output/Grid_Files/text/NTOL/")):
    os.makedirs("../../output/Grid_Files/text/NTOL/")

# Filename Tags
if (flip == False): # temporal then spatial
    tag = ("rmTM1" + str(rm_temporal_mean) + "_rmSM2" + str(rm_spatial_mean) + "_" + add_tag + "_")
else: # spatial then temporal
    tag = ("rmSM1" + str(rm_spatial_mean) + "_rmTM2" + str(rm_temporal_mean) + "_" + add_tag + "_")

# Determine Ordinal Dates for Temporal Mean Calculation
mydate1 = datetime.datetime(start_year_tm, start_month_tm, start_day_tm,12,00,00) #start_date = datetime.date.toordinal(mydate1)
mydate2 = datetime.datetime(end_year_tm, end_month_tm, end_day_tm,12,00,00) #end_date = datetime.date.toordinal(mydate2)
# Determine Date Range (From Start to End, Increasing by 24 Hours)
delta = datetime.timedelta(hours=24)
curr = mydate1
date_list = []
date_list.append(curr)
while curr < mydate2:
    curr += delta
    date_list.append(curr)

# Determine Ordinal Dates for Output
mydate1 = datetime.datetime(start_year_out, start_month_out, start_day_out,12,00,00)
mydate2 = datetime.datetime(end_year_out, end_month_out, end_day_out,12,00,00)
# Determine Date Range (From Start to End, Increasing by 24 Hours)
delta = datetime.timedelta(hours=24)
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
to_mask = np.empty((1440*720,len(date_list)))
ntol_amp = np.empty((1440*720,len(date_list)))
dates_to_delete = []
# SHAPE OF ARRAY
ntol_shape = ntol_amp.shape
# Loop Through Dates
for ii in range(0,len(date_list)):
    mydt = date_list[ii] 
    string_date = mydt.strftime('%Y%m%d') # Convert Date to String in YYYY-MM-DD Format
    print(':: Reading %s' %(string_date))
    # Complete Pathname to Current PHIBOT File
    loadfile = phibot_directory + "PHIBOT.1440x720." + string_date + ".nc"
    # Read the File
    if (os.path.isfile(loadfile)):
        llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr,cmask = read_ecco2_phibot.main(loadfile)
        #plt.imshow(amp2darr)
        #plt.show()
        # Identify and Save the Indices of the to-be Masked Values
        to_mask[:,ii] = cmask.flatten()
        # Combine Amplitude and Phase
        ntol_amp[:,ii] = np.multiply(amp,np.cos(np.radians(pha)))
    else: # File Does Not Exist
        print(':: Warning: File Does Not Exist.')
        # Save date_list index, then continue
        dates_to_delete.append(ii)
        continue

# Delete dates with no data
date_list = np.delete(date_list,dates_to_delete)
string_dates = np.delete(string_dates,dates_to_delete)
ntol_amp = np.delete(ntol_amp,dates_to_delete,axis=1)
to_mask = np.delete(to_mask,dates_to_delete,axis=1)

# Order of Removing the Temporal and Spatial Means
if (flip == False): # Temporal then Spatial
 
    # COMPUTE TEMPORAL MEAN
    if (rm_temporal_mean == True):
        ntol_temporal_avg = np.average(ntol_amp,axis=1)
        print(':: Computing temporal mean.')
        # Put Averages into Array | Efficient, but Memory Problems for lots of Dates ...
        #ntol_temporal_avg_array = np.tile(ntol_temporal_avg,(ntol_shape[1],1)).T
        # Subtract Temporal Array
        for jj in range(0,len(date_list)): # Loop Saves on Memory for Lots of Dates
            #ntol_amp[:,jj] = np.subtract(ntol_amp[:,jj], ntol_temporal_avg_array[:,jj])
            ntol_amp[:,jj] = np.subtract(ntol_amp[:,jj], ntol_temporal_avg)
    ntol_temporal_avg_array = ntol_temporal_avg = None

    # COMPUTE SPATIAL MEAN
    if (rm_spatial_mean == True):
        # Mask the Array when Computing Spatial Averages
        masked_amp = np.ma.masked_where(to_mask == 1,ntol_amp)
        ntol_spatial_avg = np.ma.average(masked_amp,axis=0)
        # Convert Back to Numpy Array
        ntol_spatial_avg = np.ma.filled(ntol_spatial_avg,fill_value=0.)
        print(':: Computing spatial mean.')
        # Put Averages into Array | Efficient, but Memory Problems for lots of Dates ...
        #ntol_spatial_avg_array = np.tile(ntol_spatial_avg,(ntol_shape[0],1))
        # Subtract Spatial Array
        for kk in range(0,len(date_list)): # Loop Saves on Memory for Lots of Dates
            #ntol_amp[:,kk] = np.subtract(ntol_amp[:,kk], ntol_spatial_avg_array[:,kk])
            ntol_amp[:,kk] = np.subtract(ntol_amp[:,kk], ntol_spatial_avg[kk])
    ntol_spatial_avg_array = ntol_spatial_avg = None

else: # spatial then temporal

    # COMPUTE SPATIAL MEAN
    if (rm_spatial_mean == True):
        # Mask the Array when Computing Spatial Averages
        masked_amp = np.ma.masked_where(to_mask == 1,ntol_amp)
        ntol_spatial_avg = np.ma.average(masked_amp,axis=0)
        # Convert Back to Numpy Array
        ntol_spatial_avg = np.ma.filled(ntol_spatial_avg,fill_value=0.)
        print(':: Computing spatial mean.')
        # Put Averages into Array | Efficient, but Memory Problems for lots of Dates ...
        #ntol_spatial_avg_array = np.tile(ntol_spatial_avg,(ntol_shape[0],1))
        # Subtract Spatial Array
        for kk in range(0,len(date_list)): # Loop Saves on Memory for Lots of Dates
            #ntol_amp[:,kk] = np.subtract(ntol_amp[:,kk], ntol_spatial_avg_array[:,kk])
            ntol_amp[:,kk] = np.subtract(ntol_amp[:,kk], ntol_spatial_avg[kk])
    ntol_spatial_avg_array = ntol_spatial_avg = None

    # COMPUTE TEMPORAL MEAN
    if (rm_temporal_mean == True):
        ntol_temporal_avg = np.average(ntol_amp,axis=1)
        print(':: Computing temporal mean.')
        # Put Averages into Array | Efficient, but Memory Problems for lots of Dates ...
        #ntol_temporal_avg_array = np.tile(ntol_temporal_avg,(ntol_shape[1],1)).T
        # Subtract Temporal Array
        for jj in range(0,len(date_list)): # Loop Saves on Memory for Lots of Dates
            #ntol_amp[:,jj] = np.subtract(ntol_amp[:,jj], ntol_temporal_avg_array[:,jj])
            ntol_amp[:,jj] = np.subtract(ntol_amp[:,jj], ntol_temporal_avg)
    ntol_temporal_avg_array = ntol_temporal_avg = None

# Convert to Masked Array (Re-set all masked grid points to zero)
masked_amp = np.ma.masked_where(to_mask == 1,ntol_amp)
print(':: Masking the Amplitude Array.')
for bb in range(0,len(date_list)): # Loop Saves on Memory for Lots of Dates
    ntol_amp[:,bb] = np.ma.filled(masked_amp[:,bb],fill_value=0.)
masked_amp = to_mask = None

# Set all Phase Values to Zero (Amps are Positive and Negative)
ntol_pha = np.zeros((1440*720,len(date_list)))

# Loop Through Dates and Write to File
for kk in range(0,len(date_list_out)):
    
    # Output NTOL Grid to File for Plotting in GMT
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
    print(':: Writing %s' %(string_date))
    ntol_out = ("ntol_" + tag + string_date + ".txt")
    ntol_out_nc = ("ntol_" + tag + string_date + ".nc")
    ntol_file = ("../../output/Grid_Files/GMT/NTOL/height-anomaly_" + ntol_out)
    ntol_file_pressure = ("../../output/Grid_Files/GMT/NTOL/pressure_" + ntol_out)
    ntol_file_nc = ("../../output/Grid_Files/nc/NTOL/convgf_" + ntol_out_nc)
    ntol_file_text = ("../../output/Grid_Files/text/NTOL/convgf_" + ntol_out)
    # Prepare Data
    all_ntol_data = np.column_stack((llon,llat,ntol_amp[:,jj]))
    all_ntol_data_pressure = np.column_stack((llon,llat,ntol_amp[:,jj]*9.81*1027.5)) # PHIBOT calibrated by 1027.5 in ECCO2
    all_ntol_data_convgf = np.column_stack((llat,llon,ntol_amp[:,jj],ntol_pha[:,jj]))
    # Write Files
    if (write_nc == True):
        print(":: Writing netCDF-formatted file.")
        # Open new NetCDF file in "write" mode
        dataset = netCDF4.Dataset(ntol_file_nc,'w',format='NETCDF4_CLASSIC')
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
        amplitudes[:] = ntol_amp[:,jj]
        phases[:] = ntol_pha[:,jj]
        # Write Data to File
        dataset.close()
    if (write_gmt == True):
        print(":: Writing GMT-convenient text file.")
        np.savetxt(ntol_file, all_ntol_data, fmt='%f %f %f')
        np.savetxt(ntol_file_pressure, all_ntol_data_pressure, fmt='%f %f %f')
    if (write_txt == True):
        print(":: Writing plain-text file.")
        np.savetxt(ntol_file_text, all_ntol_data_convgf, fmt='%f %f %f %f')

if (write_gmt == True):

    print(":: Writing to GMT-convenient text file.") 
    # Compute Standard Deviation for Each Amplitude Pixel
    ntol_std = np.std(ntol_amp,axis=1)
    # Export Standard Deviation to File
    ntol_out = "ntol_std_" + string_dates[0] + "_" + string_dates[-1] + ".txt"
    ntol_file = ("../../output/Grid_Files/GMT/NTOL/height-anomaly_" + ntol_out)
    ntol_file_pressure = ("../../output/Grid_Files/GMT/NTOL/pressure_" + ntol_out)
    # Prepare Data
    all_ntol_data = np.column_stack((llon,llat,ntol_std))
    all_ntol_data_pressure = np.column_stack((llon,llat,ntol_std*9.81*1027.5))  # PHIBOT calibrated by 1027.5 in ECCO2
    # Write Data to File
    np.savetxt(ntol_file, all_ntol_data, fmt='%f %f %f')
    np.savetxt(ntol_file_pressure, all_ntol_data_pressure, fmt='%f %f %f')

    # Compute Maximum for Each Amplitude Pixel
    ntol_abs = np.absolute(ntol_amp)
    ntol_max = np.amax(ntol_abs,axis=1)
    # Export Standard Deviation to File
    ntol_out = "ntol_max_" + string_dates[0] + "_" + string_dates[-1] + ".txt"
    ntol_file = ("../../output/Grid_Files/GMT/NTOL/height-anomaly_" + ntol_out)
    ntol_file_pressure = ("../../output/Grid_Files/GMT/NTOL/pressure_" + ntol_out)
    # Prepare Data
    all_ntol_data = np.column_stack((llon,llat,ntol_max))
    all_ntol_data_pressure = np.column_stack((llon,llat,ntol_max*9.81*1027.5))  # PHIBOT calibrated by 1027.5 in ECCO2
    # Write Data to File
    np.savetxt(ntol_file, all_ntol_data, fmt='%f %f %f')
    np.savetxt(ntol_file_pressure, all_ntol_data_pressure, fmt='%f %f %f')

# --------------------- END CODE --------------------------- #


