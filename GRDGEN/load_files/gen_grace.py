#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO GENERATE GRIDS FOR LOADS FROM GRACE
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
from GRDGEN.utility import read_grace

# --------------- SPECIFY USER INPUTS --------------------- #

# Atmospheric Surface Pressure Files from GRACE - MUST HAVE NETCDF4 FOR PYTHON INSTALLED 
#  Specify the directory containing the yearly netcdf files here:
grace_directory = ("../../input/Load_Models/GRACE/data/")

# Date Range for Temporal-Mean Computation (yyyy, mm, dd); End Day is Included (Files to be Read in)
start_year_tm = 2010; start_month_tm = 1; start_day_tm = 1
end_year_tm = 2017; end_month_tm = 1; end_day_tm = 1

# Date Range for Output Files (yyyy, mm, dd); End Day is Included (Files to be Written out)
start_year_out = 2010; start_month_out = 1; start_day_out = 1
end_year_out = 2017; end_month_out = 1; end_day_out = 1
 
# Remove spatial and temporal averages?
rm_spatial_mean = False
rm_temporal_mean = False

# Order in which to remove the temporal (t) and spatial (s) averages (false = t then s; true = s then t)
flip = False

# Additional Name Tag
tmrange = "%4d%02d%02d-%4d%02d%02d" % (start_year_tm, start_month_tm, start_day_tm, end_year_tm, end_month_tm, end_day_tm)
add_tag = (tmrange + "_GRACE")

# Complete Pathname to Current GRACE File

# FOR first solution
loadfile1 = grace_directory + "GRCTellus.JPL.200204_201701.LND.RL05_1.DSTvSCS1411.nc"
tag1 = "JPL"

# FOR second solution
loadfile2 = grace_directory + "GRCTellus.CSR.200204_201701.LND.RL05.DSTvSCS1409.nc"
tag2 = "CSR"

# FOR third solution
loadfile3 = grace_directory + "GRCTellus.GFZ.200204_201701.LND.RL05.DSTvSCS1409.nc"
tag3 = "GFZ"

# Scaling factors
scaling = grace_directory + "CLM4.SCALE_FACTOR.DS.G300KM.RL05.DSTvSCS1409.nc"

# Average solutions
avgsol = True

# Apply GRACE scaling factors
appscl = True

# Solution tag
if (avgsol == True):
    soltag = (tag1 + "-" + tag2 + "-" + tag3)
else:
    soltag = (tag1)

# Write Load Information to a netCDF-formatted File? (Default for convolution)
write_nc = True

# Write Load Information to a Text File? (Alternative for convolution)
write_txt = False

# Write Load Information to a GMT-formatted File?
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
if not (os.path.isdir("../../output/Grid_Files/GMT/GRACE/")):
    os.makedirs("../../output/Grid_Files/GMT/GRACE/")
if not (os.path.isdir("../../output/Grid_Files/nc/")):
    os.makedirs("../../output/Grid_Files/nc/")
if not (os.path.isdir("../../output/Grid_Files/nc/GRACE/")):
    os.makedirs("../../output/Grid_Files/nc/GRACE/")
if not (os.path.isdir("../../output/Grid_Files/text/")):
    os.makedirs("../../output/Grid_Files/text/")
if not (os.path.isdir("../../output/Grid_Files/text/GRACE/")):
    os.makedirs("../../output/Grid_Files/text/GRACE/")

# Filename Tags
if (flip == False): # temporal then spatial
    tag = ("rmTM1" + str(rm_temporal_mean) + "_rmSM2" + str(rm_spatial_mean) + "_" + add_tag + "_" + soltag + "_Scaling" + str(appscl) + "_")
else: # spatial then temporal
    tag = ("rmSM1" + str(rm_spatial_mean) + "_rmTM2" + str(rm_temporal_mean) + "_" + add_tag + "_" + soltag + "_Scaling" + str(appscl) + "_")


# Determine Ordinal Dates for Temporal Mean Calculation
mydate1 = datetime.datetime(start_year_tm, start_month_tm, start_day_tm, 0, 0, 0) #start_date = datetime.date.toordinal(mydate1)
mydate2 = datetime.datetime(end_year_tm, end_month_tm, end_day_tm, 0, 0, 0) #end_date = datetime.date.toordinal(mydate2)

# Determine Date Range (From Start to End, Increasing by 1 day; sometimes GRACE has two models per month)
curr = mydate1
date_list = []
date_list.append(curr)
count = 0
while curr < mydate2:
    curr += datetime.timedelta(days=1)
    date_list.append(curr)
# Determine Date Range (From Start to End, Increasing by 30 days)
#    count = count+1
#    if ((start_month_tm + count) == 13):
#        start_year_tm += 1
#        start_month_tm = 1
#        count = 0
#    curr = datetime.datetime(start_year_tm, (start_month_tm + count), 1,0,0,0)
#    date_list.append(curr)

# Determine Ordinal Dates for Output
mydate1 = datetime.datetime(start_year_out, start_month_out, start_day_out, 0, 0, 0)
mydate2 = datetime.datetime(end_year_out, end_month_out, end_day_out, 0, 0, 0)

# Determine Date Range (From Start to End, Increasing by 1 day; sometimes GRACE has two models per month)
curr = mydate1
date_list_out = []
date_list_out.append(curr)
count = 0
while curr < mydate2:
    curr += datetime.timedelta(days=1)
    date_list_out.append(curr)
# Determine Date Range (From Start to End, Increasing by 30 days)
#    count = count+1
#    if ((start_month_out + count) == 13):
#        start_year_out += 1
#        start_month_out = 1
#        count = 0
#    curr = datetime.datetime(start_year_out, (start_month_out + count), 1,0,0,0)
#    date_list_out.append(curr)

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
to_mask = np.empty((360*180,len(date_list)))
grace_amp = np.empty((360*180,len(date_list)))
dates_to_delete = []
# SHAPE OF ARRAY
grace_shape = grace_amp.shape
# Loop Through Dates
for ii in range(0,len(date_list)):
    mydt = date_list[ii] 
    string_date = mydt.strftime('%Y%m%d%H%M%S') # Convert Date to String in YYYY-mm-dd-HH-MM-SS Format
    print(':: Reading %s' %(string_date))
    string_year = mydt.strftime('%Y') # Convert Date to String in YYYY Format
    
    # Read the File 
    if (os.path.isfile(loadfile1)):
        llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_grace.main(loadfile1,mydt,ldfl2=loadfile2,ldfl3=loadfile3,scl=scaling,avsolns=avgsol,appscaling=appscl)
        if llat is None:
            print(':: Warning: Date does not exist within file.')
            # Save date_list index, then continue
            dates_to_delete.append(ii)
            continue
            #to_mask[:,ii] = np.ones((grace_shape[0],)) # set mask to true
            #grace_amp[:,ii] = np.zeros((grace_shape[0],))
        
        # Combine Amplitude and Phase 
        else:
            to_mask[:,ii] = np.zeros((grace_shape[0],)) # file exists : do not apply mask
            grace_amp[:,ii] = np.multiply(amp,np.cos(np.radians(pha)))
            lat_array = llat.copy()
            lon_array = llon.copy()
        
    else: # File Does Not Exist
        print(':: Warning: Load File Does Not Exist.')
        #to_mask[:,ii] = np.ones((grace_shape[0],)) # set mask to true
        #grace_amp[:,ii] = np.zeros((grace_shape[0],))

# Delete dates with no data
date_list = np.delete(date_list,dates_to_delete)
string_dates = np.delete(string_dates,dates_to_delete)
grace_amp = np.delete(grace_amp,dates_to_delete,axis=1)
to_mask = np.delete(to_mask,dates_to_delete,axis=1)
llat = lat_array.copy(); llon = lon_array.copy()

# Order of Removing the Temporal and Spatial Means
if (flip == False): # Temporal then Spatial
 
    # COMPUTE TEMPORAL MEAN
    if (rm_temporal_mean == True):
        grace_temporal_avg = np.average(grace_amp,axis=1)
        print(('Maximum amplitude (meters): ', np.max(grace_amp)))
        print(':: Computing temporal mean.')
        # Put Averages into Array | Efficient, but Memory Problems for lots of Dates ...
        #grace_temporal_avg_array = np.tile(grace_temporal_avg,(grace_shape[1],1)).T
        # Subtract Temporal Array
        for jj in range(0,len(date_list)): # Loop Takes Longer, but Saves on Memory
            #grace_amp[:,jj] = np.subtract(grace_amp[:,jj], grace_temporal_avg_array[:,jj])
            #print(max(grace_amp[:,jj]))            
            grace_amp[:,jj] = np.subtract(grace_amp[:,jj], grace_temporal_avg)
            #print(max(grace_amp[:,jj])) 
        #print(grace_temporal_avg)
        #print(grace_amp)
        
    grace_temporal_avg_array = grace_temporal_avg = None

    # COMPUTE SPATIAL MEAN
    if (rm_spatial_mean == True):
        # Mask the Array when Computing Spatial Averages
        masked_amp = np.ma.masked_where(to_mask == 1,grace_amp)
        grace_spatial_avg = np.ma.average(masked_amp,axis=0)
        # Convert Back to Numpy Array
        grace_spatial_avg = np.ma.filled(grace_spatial_avg,fill_value=0.)
        print(('Maximum amplitude (meters): ', np.max(grace_amp)))
        print(':: Computing spatial mean.')
        # Put Averages into Array | Efficient, but Memory Problems for lots of Dates ...
        #grace_spatial_avg_array = np.tile(grace_spatial_avg,(grace_shape[0],1))
        # Subtract Spatial Array
        for kk in range(0,len(date_list)): # Loop Takes Longer, but Saves on Memory
            #grace_amp[:,kk] = np.subtract(grace_amp[:,kk], grace_spatial_avg_array[:,kk])
            grace_amp[:,kk] = np.subtract(grace_amp[:,kk], grace_spatial_avg[kk])
    grace_spatial_avg_array = grace_spatial_avg = None

else: # Spatial then Temporal

    # COMPUTE SPATIAL MEAN
    if (rm_spatial_mean == True):
        # Mask the Array when Computing Spatial Averages
        masked_amp = np.ma.masked_where(to_mask == 1,grace_amp)
        grace_spatial_avg = np.ma.average(masked_amp,axis=0)
        # Convert Back to Numpy Array
        grace_spatial_avg = np.ma.filled(grace_spatial_avg,fill_value=0.)
        print(':: Computing spatial mean.')
        # Put Averages into Array | Efficient, but Memory Problems for lots of Dates ...
        #grace_spatial_avg_array = np.tile(grace_spatial_avg,(grace_shape[0],1))
        # Subtract Spatial Array
        for kk in range(0,len(date_list)): # Loop Takes Longer, but Saves on Memory
            #grace_amp[:,kk] = np.subtract(grace_amp[:,kk], grace_spatial_avg_array[:,kk])
            grace_amp[:,kk] = np.subtract(grace_amp[:,kk], grace_spatial_avg[kk])
    grace_spatial_avg_array = grace_spatial_avg = None

    # COMPUTE TEMPORAL MEAN
    if (rm_temporal_mean == True):
        grace_temporal_avg = np.average(grace_amp,axis=1)
        print(':: Computing temporal mean.')
        # Put Averages into Array
        #grace_temporal_avg_array = np.tile(grace_temporal_avg,(grace_shape[1],1)).T
        # Subtract Temporal Array | Efficient, but Memory Problems for lots of Dates ...
        for jj in range(0,len(date_list)): # Loop Takes Longer, but Saves on Memory
            #grace_amp[:,jj] = np.subtract(grace_amp[:,jj], grace_temporal_avg_array[:,jj])
            grace_amp[:,jj] = np.subtract(grace_amp[:,jj], grace_temporal_avg)
    grace_temporal_avg_array = grace_temporal_avg = None

# Convert to Masked Array (Re-set all masked grid points to zero)
masked_amp = np.ma.masked_where(to_mask == 1,grace_amp)
print(':: Masking the amplitude array.')
for bb in range(0,len(date_list)): # Loop Takes Longer, but Saves on Memory
    grace_amp[:,bb] = np.ma.filled(masked_amp[:,bb],fill_value=0.)
    #print(np.max(grace_amp[:,bb]))
masked_amp = to_mask = None

# Set Phase to Zero (Amplitudes Contain Phase)
grace_pha = np.zeros((360*180,len(date_list)))
print(llon)
print(llat)
print(grace_amp)

# Loop Through Dates and Write to File
for kk in range(0,len(date_list_out)):
    
    # Output ATML Grid to File for Plotting in GMT
    mydt = date_list_out[kk]
    # Convert Date to String in YYYY-mm-dd-HH-MM-SS Format
    string_date = mydt.strftime('%Y%m%d%H%M%S')
    # Locate Date in Full Date List
    diff_dates = []
    diff_dates_sec = []
    for ii in range(0,len(date_list)):
        # Find Grace file that matches year, month, and day
        diff_dates.append(date_list[ii]-mydt)
        diff_dates_sec.append(datetime.timedelta.total_seconds(diff_dates[ii]))
    diff_dates_sec = np.asarray(diff_dates_sec)
    dselect = np.where(diff_dates_sec == 0.0); dselect = dselect[0]
    if (len(dselect > 0)):
        dselect = dselect[0]
    else:
        print(':: Warning: No Date Match Found for Output Date in Range of Amplitude Array | %s' %(string_date))
        continue
    #idx = str(jj) # Convert to string to test if a value exists (including zero)
    #if not idx:
    #    print(':: Warning: No Date Match Found for Output Date in Range of Amplitude Array | %s' %(string_date))
    #    continue
    #jj = int(idx) # Convert back to integer to use as index
    # Prepare to Write to File
    print(':: Writing %s' %(string_dates[dselect]))
    grace_out = ("grace_" + tag + string_date + ".txt")
    grace_out_nc = ("grace_" + tag + string_date + ".nc")
    grace_file = ("../../output/Grid_Files/GMT/GRACE/height-anomaly_" + grace_out)
    grace_file_pressure = ("../../output/Grid_Files/GMT/GRACE/pressure_" + grace_out)
    grace_file_nc = ("../../output/Grid_Files/nc/GRACE/convgf_" + grace_out_nc)
    grace_file_text = ("../../output/Grid_Files/text/GRACE/convgf_" + grace_out)
    # Prepare Data
    all_grace_data = np.column_stack((llon,llat,grace_amp[:,dselect]))
    all_grace_data_pressure = np.column_stack((llon,llat,grace_amp[:,dselect]*9.81))
    all_grace_data_convgf = np.column_stack((llat,llon,grace_amp[:,dselect],grace_pha[:,dselect]))
    # Write Files
    if (write_nc == True):
        print(":: Writing netCDF-formatted file.")
        # Open new NetCDF file in "write" mode
        dataset = netCDF4.Dataset(grace_file_nc,'w',format='NETCDF4_CLASSIC')
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
        amplitudes[:] = grace_amp[:,dselect]
        phases[:] = grace_pha[:,dselect]
        # Write Data to File
        dataset.close()
    if (write_gmt == True):
        print(":: Writing to GMT-convenient text file.")
        np.savetxt(grace_file, all_grace_data, fmt='%f %f %f')
        np.savetxt(grace_file_pressure, all_grace_data_pressure, fmt='%f %f %f')
    if (write_txt == True):
        print(":: Writing to plain-text file.")
        np.savetxt(grace_file_text, all_grace_data_convgf, fmt='%f %f %f %f')

if (write_gmt == True):

    print(":: Writing to GMT-convenient text file.")
    # Compute Standard Deviation for Each Amplitude Pixel
    grace_std = np.std(grace_amp,axis=1)
    # Export Standard Deviation to File
    grace_out = "grace_std_" + string_dates[0] + "_" + string_dates[-1] + ".txt"
    grace_file = ("../../output/Grid_Files/GMT/GRACE/height-anomaly_" + grace_out)
    grace_file_pressure = ("../../output/Grid_Files/GMT/GRACE/pressure_" + grace_out)
    # Prepare Data
    #grace_std = np.ma.filled(grace_std,fill_value=0.)
    all_grace_data = np.column_stack((llon,llat,grace_std))
    all_grace_data_pressure = np.column_stack((llon,llat,grace_std*9.81))
    # Write Data to File
    np.savetxt(grace_file, all_grace_data, fmt='%f %f %f')
    np.savetxt(grace_file_pressure, all_grace_data_pressure, fmt='%f %f %f')

    # Compute Maximum for Each Amplitude Pixel
    grace_abs = np.absolute(grace_amp)
    grace_max = np.amax(grace_abs,axis=1)
    # Export Standard Deviation to File
    grace_out = "grace_max_" + string_dates[0] + "_" + string_dates[-1] + ".txt"
    grace_file = ("../../output/Grid_Files/GMT/GRACE/height-anomaly_" + grace_out)
    grace_file_pressure = ("../../output/Grid_Files/GMT/GRACE/pressure_" + grace_out)
    # Prepare Data
    all_grace_data = np.column_stack((llon,llat,grace_max))
    all_grace_data_pressure = np.column_stack((llon,llat,grace_max*9.81))
    # Write Data to File
    np.savetxt(grace_file, all_grace_data, fmt='%f %f %f')
    np.savetxt(grace_file_pressure, all_grace_data_pressure, fmt='%f %f %f')

# --------------------- END CODE --------------------------- #


