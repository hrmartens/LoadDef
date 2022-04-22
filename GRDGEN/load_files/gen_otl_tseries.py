#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO GENERATE GRIDS FOR OCEANIC TIDAL LOADING AS A TIME SERIES
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

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd() + "/../../")

# IMPORT PYTHON MODULES
import numpy as np
import scipy as sc
import datetime
import netCDF4
from GRDGEN.utility import read_fes2004
from GRDGEN.utility import read_fes2012
from GRDGEN.utility import read_fes2014
from GRDGEN.utility import read_tpxo9atlas
from GRDGEN.utility import read_tpxo8atlas
from GRDGEN.utility import read_tpxo7atlas
from GRDGEN.utility import read_got410c
from GRDGEN.utility import read_got410
from GRDGEN.utility import read_eot11a
from GRDGEN.utility import read_hamtide11a
from GRDGEN.utility import read_osu12
from GRDGEN.utility import read_osu_local
from GRDGEN.utility import read_schwiderski
from GRDGEN.utility import read_dtu10
from CONVGF.utility import read_AmpPha

# --------------- SPECIFY USER INPUTS --------------------- #

# 1. Specify the full path to the desired tidal model: Examples Provided
loadfile = ("../../input/Load_Models/FES2014b/m2.nc")
#loadfile = ("../../input/Load_Models/GOT4.10c/got410c.m2.dat")
 
# 2. Specify the type of loading model
ftype = 12 # 1=FES2012, 2=FES2004, 3=TPXO8-Atlas, 4=GOT4.10c, 5=EOT11A, 6 = HAMTIDE11A, 7=OSU12, 8=LOCAL, 9=SCHWIDERSKI, 10=DTU10, 11=GOT4.10, 12=FES2014b, 13=TPXO7-Atlas, 14=TPXO9-Atlas 
 
# 3. Specify the tidal harmonic
harmonic = ("M2")

# 4. Specify period of the tidal harmonic (hours)
period = 12.42

# 5. Specify the number of files to generate (number of epochs) for a single tidal period
num_epochs = 50

# 6. Write Load Information to a netCDF4-formatted File? (Default for convolution)
write_nc = True

# 7. Write Load Information to a Text File? (Alternative for convolution)
write_txt = False
 
# 8. Write Load Information to a GMT-formatted File? (Lon, Lat, Amplitude)
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
if not (os.path.isdir("../../output/Grid_Files/GMT/OTL/")):
    os.makedirs("../../output/Grid_Files/GMT/OTL/")
if not (os.path.isdir("../../output/Grid_Files/nc/")):
    os.makedirs("../../output/Grid_Files/nc/")
if not (os.path.isdir("../../output/Grid_Files/nc/OTL/")):
    os.makedirs("../../output/Grid_Files/nc/OTL/")
if not (os.path.isdir("../../output/Grid_Files/text/")):
    os.makedirs("../../output/Grid_Files/text/")
if not (os.path.isdir("../../output/Grid_Files/text/OTL/")):
    os.makedirs("../../output/Grid_Files/text/OTL/")

# Read Load File (Should Return 1D Arrays - Lon,Lat,Amp,Pha; 
#  as well as original format - vectors and matrices)
print(":: Reading model. Please wait.")
if (ftype == 1):
    llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_fes2012.main(loadfile)
    model = ("FES2012")
elif (ftype == 2):
    llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_fes2004.main(loadfile)
    model = ("FES2004")
elif (ftype == 3):
    llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_tpx08atlas.main(loadfile)
    model = ("TPXO8-Atlas")
elif (ftype == 4):
    llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_got410c.main(loadfile)
    model = ("GOT410c")
elif (ftype == 5):
    llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_eot11a.main(loadfile)
    model = ("EOT11A")
elif (ftype == 6):
    llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_hamtide11a.main(loadfile)
    model = ("HAMTIDE11A")
elif (ftype == 7):
    llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_osu12.main(loadfile)
    model = ("OSU12")
elif (ftype == 8):
    llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_osu_local.main(loadfile)
    model = ("Local")
elif (ftype == 9):
    llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_schwiderski.main(loadfile)
    model = ("Schwiderski")
elif (ftype == 10):
    llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_dtu10.main(loadfile)
    model = ("DTU10")
elif (ftype == 11):
    llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_got410.main(loadfile)
    model = ("GOT410")
elif (ftype == 12):
    llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_fes2014.main(loadfile)
    model = ("FES2014")
elif (ftype == 13):
    llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_tpxo7atlas.main(loadfile,harmonic)
    model = ("TPXO7-Atlas")
elif (ftype == 14):
    llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_tpxo9atlas.main(loadfile)
    model = ("TPXO9-Atlas")
    print('Model %s is a high-resolution global grid, and it may take several minutes to generate the output files. Please wait...' %(model))
else:
    print("Error: Incorrect ftype for Loading Model")
    sys.exit()

## Force to Constant Value Everywhere
##amp = np.zeros((len(amp),)) + 10.0
##pha = np.zeros((len(pha),)) 

# Compute Period (in seconds) and Frequency of Tidal Harmonic 
per_sec = period * (60 * 60)
freq_deg_per_hour = 360 / period
freq_hz = freq_deg_per_hour / (60 * 60)

# Generate Time Epochs
time = np.linspace(0.0,period,num=num_epochs,endpoint=True)

# Loop through the Time Epochs
for jj in range(0,len(time)):

    # Current Time
    myt = time[jj] 
    myt_sec = myt * (60*60)
    myt_str = "{0:.2f}".format(round(myt,2))
    print(myt_str)

    # LOAD MODEL
    arg = np.radians(freq_hz * myt_sec - pha)
    otl = np.multiply(amp,np.cos(arg))
    otl_pha = np.zeros(len(amp))
 
    # Output OTL Grid to File for Plotting in GMT
    otl_out = model + "-" + harmonic + "-" + myt_str + ".txt"
    otl_out_nc = model + "-" + harmonic + "-" + myt_str + ".nc"
    otl_file_gmt = ("../../output/Grid_Files/GMT/OTL/height-anomaly_" + otl_out)
    otl_file_pressure = ("../../output/Grid_Files/GMT/OTL/pressure_" + otl_out)
    otl_file_nc = ("../../output/Grid_Files/nc/OTL/convgf_" + otl_out_nc)
    otl_file_text = ("../../output/Grid_Files/text/OTL/convgf_" + otl_out)
    # Prepare Data
    all_otl_data_gmt = np.column_stack((llon,llat,otl))
    all_otl_data_pressure = np.column_stack((llon,llat,otl*9.81*1030.0))
    all_otl_data_convgf = np.column_stack((llat,llon,otl,otl_pha))
    # Write Files
    if (write_nc == True):
        print(":: Writing netCDF-formatted file.")
        # Open new NetCDF file in "write" mode
        dataset = netCDF4.Dataset(otl_file_nc,'w',format='NETCDF4_CLASSIC')
        # Define dimensions for variables
        num_pts = len(llat)
        latitude = dataset.createDimension('latitude',num_pts)
        longitude = dataset.createDimension('longitude',num_pts)
        amplitude = dataset.createDimension('amplitude',num_pts)
        phase = dataset.createDimension('phase',num_pts)
        # Create variables
        latitudes = dataset.createVariable('latitude',float,('latitude',))
        longitudes = dataset.createVariable('longitude',float,('longitude',))
        amplitudes = dataset.createVariable('amplitude',float,('amplitude',))
        phases = dataset.createVariable('phase',float,('phase',))
        # Add units
        latitudes.units = 'degree_north'
        longitudes.units = 'degree_east'
        amplitudes.units = 'm'
        phases.units = 'degree'
        # Assign data
        latitudes[:] = llat
        longitudes[:] = llon
        amplitudes[:] = otl
        phases[:] = otl_pha
        # Write Data to File
        dataset.close()
    if (write_gmt == True):
        print(":: Writing GMT-convenient text file.")
        np.savetxt(otl_file_gmt, all_otl_data_gmt, fmt='%f %f %f')
        np.savetxt(otl_file_pressure, all_otl_data_pressure, fmt='%f %f %f')
    if (write_txt == True):
        print(":: Writing plain-text file.")
        np.savetxt(otl_file_text, all_otl_data_convgf, fmt='%f %f %f %f')
 
# Output OTL Grid to File for Use with LoadDef (amp and pha; not snapshots in time)
if (write_nc == True):
    print(":: Writing netCDF-formatted file.")
    otl_out = (model + "-" + harmonic + ".nc")
    otl_file = ("../../output/Grid_Files/nc/OTL/convgf_" + otl_out)
    # Open new NetCDF file in "write" mode
    dataset = netCDF4.Dataset(otl_file,'w',format='NETCDF4_CLASSIC')
    # Define dimensions for variables
    num_pts = len(llat)
    latitude = dataset.createDimension('latitude',num_pts)
    longitude = dataset.createDimension('longitude',num_pts)
    amplitude = dataset.createDimension('amplitude',num_pts)
    phase = dataset.createDimension('phase',num_pts)
    # Create variables
    latitudes = dataset.createVariable('latitude',float,('latitude',))
    longitudes = dataset.createVariable('longitude',float,('longitude',))
    amplitudes = dataset.createVariable('amplitude',float,('amplitude',))
    phases = dataset.createVariable('phase',float,('phase',))
    # Add units
    latitudes.units = 'degree_north'
    longitudes.units = 'degree_east'
    amplitudes.units = 'm'
    phases.units = 'degree'
    # Assign data
    latitudes[:] = llat
    longitudes[:] = llon
    amplitudes[:] = amp
    phases[:] = pha
    # Write Data to File
    dataset.close()
if (write_txt == True):
    print(":: Writing plain-text file.")
    otl_out = (model + "-" + harmonic + ".txt")
    otl_file = ("../../output/Grid_Files/text/OTL/convgf_" + otl_out)
    # Prepare Data
    all_otl_data = np.column_stack((llat,llon,amp,pha))
    # Write Data to File
    np.savetxt(otl_file, all_otl_data, fmt='%f %f %f %f')

# --------------------- END CODE --------------------------- #


