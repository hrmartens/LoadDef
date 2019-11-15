#!/usr/bin/env python

# *********************************************************************
# MAIN PROGRAM TO PREDICT SURFACE DISPLACEMENTS CAUSED BY SURFACE MASS LOADING 
# BY CONVOLVING DISPLACEMENT LOAD GREENS FUNCTIONS WITH A MODEL FOR A SURFACE MASS LOAD 
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

# IMPORT PRINT FUNCTION
from __future__ import print_function

# IMPORT MPI MODULE
from mpi4py import MPI

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
import sys
import os
sys.path.append(os.getcwd() + "/../")

# IMPORT PYTHON MODULES
import numpy as np
import scipy as sc
import datetime
from CONVGF.CN import load_convolution
from CONVGF.utility import read_station_file

# --------------- SPECIFY USER INPUTS --------------------- #

# Reference Frame (used for filenames) [Blewitt 2003]
rfm = "cm"

# Greens Function File
#  :: May be load Green's function file output directly from run_gf.py (norm_flag = False)
#  :: May be from a published table, normalized according to Farrell (1972) conventions [theta, u_norm, v_norm]
grn_file = ("../output/Greens_Functions/" + rfm + "_PREM.txt")
norm_flag  = False

# Full Path to Load Directory and Prefix of Filename
loadfile_directory = ("../output/Grid_Files/nc/OTL/")  # Example 1 (ocean tidal loading)
#loadfile_directory = ("../output/Grid_Files/nc/NTOL/")  # Example 2 (time series)

# Prefix for the Load Files (Load Directory will be Searched for all Files Starting with this Prefix)
#  :: Note: For Load Files Organized by Date, the End of Filename Name Must be in the Format yyyymmddhhmnsc.txt
#  :: Note: If not organized by date, files may be organized by tidal harmonic, for example (i.e. a unique filename ending)
#  :: Note: Output names (within output files) will be determined by extension following last underscore character (e.g., date/harmonic/model)
loadfile_prefix = ("convgf_TPXO8-Atlas") # Example 1 (ocean tidal loading)
#loadfile_prefix = ("convgf_ntol") # Example 2 (time series)

# LoadFile Format: ["nc", "txt"]
loadfile_format = "nc"
 
# Are the Load Files Organized by Datetime?
#  :: If False, all Files that match the loadfile directory and prefix will be analyzed.
time_series = False  # Example 1 (ocean tidal loading)
#time_series = True  # Example 2 (time series)

# Date Range for Computation (Year,Month,Day,Hour,Minute,Second)
#  :: Note: Only used if 'time_series' is True
frst_date = [2015,1,1,0,0,0]
last_date = [2016,3,1,0,0,0]

# Are the load values on regular grids (speeds up interpolation); If unsure, leave as false.
regular = True

# Load Density
#  Recommended: 1025-1035 for oceanic loads (e.g., FES2014, ECCO2); 1 for atmospheric loads (e.g. ECMWF)
ldens = 1030.0
  
# Ocean/Land Mask 
#  :: 0 = do not mask ocean or land (retain full model); 1 = mask out land (retain ocean); 2 = mask out oceans (retain land)
#  :: Recommended: 1 for oceanic; 2 for atmospheric
lsmask_type = 1

# Full Path to Land-Sea Mask File (May be Irregular and Sparse)
#  :: Format: Lat, Lon, Mask [0=ocean; 1=land]
lsmask_file = ("../input/Land_Sea/ETOPO1_Ice_g_gmt4_wADD.txt")

# Station/Grid-Point Location File (Lat, Lon, StationName)
sta_file = ("../input/Station_Locations/PBO_Select.txt")

# Optional: Additional string to include in output filenames (e.g. "_2019")
outstr = ("")

# ------------------ END USER INPUTS ----------------------- #

# -------------------- SETUP MPI --------------------------- #

# Get the Main MPI Communicator That Controls Communication Between Processors
comm = MPI.COMM_WORLD
# Get My "Rank", i.e. the Processor Number Assigned to Me
rank = comm.Get_rank()
# Get the Total Number of Other Processors Used
size = comm.Get_size()

# ---------------------------------------------------------- #

# -------------------- BEGIN CODE -------------------------- #

# Ensure that the Output Directories Exist
if (rank == 0):
    if not (os.path.isdir("../output/Convolution/")):
        os.makedirs("../output/Convolution/")

# Check format of load files
if not (loadfile_format == "nc"):
    if not (loadfile_format == "txt"):
        print(":: Error: Invalid format for load files. See scripts in the /GRDGEN/load_files/ folder. Acceptable formats: netCDF, txt.")
 
# Read Station & Date Range File
lat,lon,sta = read_station_file.main(sta_file)

# Determine Number of Stations Read In
if isinstance(lat,float) == True: # only 1 station
    numel = 1
else:
    numel = len(lat)

# Loop Through Each Station
for jj in range(0,numel):

    # Remove Index If Only 1 Station
    if (numel == 1): # only 1 station read in
        my_sta = sta
        my_lat = lat
        my_lon = lon
    else:
        my_sta = sta[jj]
        my_lat = lat[jj]
        my_lon = lon[jj]

    # If Rank is Master, Output Station Name
    my_sta = my_sta.decode()
    if (rank == 0):
        print(' ')
        print(':: Starting on Station: ' + my_sta)

    # Output File Name
    cnv_out = my_sta + "_" + rfm + "_" + loadfile_prefix + outstr + ".txt"

    # Convert Start and End Dates to Datetimes
    if (time_series == True):
        frstdt = datetime.datetime(frst_date[0],frst_date[1],frst_date[2],frst_date[3],frst_date[4],frst_date[5])
        lastdt = datetime.datetime(last_date[0],last_date[1],last_date[2],last_date[3],last_date[4],last_date[5])

    # Determine Number of Matching Load Files
    load_files = []
    if os.path.isdir(loadfile_directory):
        for mfile in os.listdir(loadfile_directory): # Filter by Load Directory
            if mfile.startswith(loadfile_prefix): # Filter by File Prefix
                if (time_series == True):
                    if (loadfile_format == "txt"):
                        mydt = datetime.datetime.strptime(mfile[-18:-4],'%Y%m%d%H%M%S') # Convert Filename String to Datetime
                    elif (loadfile_format == "nc"):
                        mydt = datetime.datetime.strptime(mfile[-17:-3],'%Y%m%d%H%M%S') # Convert Filename String to Datetime
                    else:
                        print(":: Error: Invalid format for load files. See scripts in the /GRDGEN/load_files/ folder. Acceptable formats: netCDF, txt.")
                    if ((mydt >= frstdt) & (mydt <= lastdt)): # Filter by Date Range
                        load_files.append(loadfile_directory + mfile) # Append File to List
                else:
                    load_files.append(loadfile_directory + mfile) # Append File to List
    else:
        sys.exit('Error: The loadfile directory does not exist. You may need to create it. The /GRDGEN/load_files/ folder contains utility scripts to convert common models into LoadDef-compatible formats, and will automatically create a loadfile directory.')

    # Test for Load Files
    if not load_files:
        sys.exit('Error: Could not find load files. You may need to generate them. The /GRDGEN/load_files/ folder contains utility scripts to convert common models into LoadDef-compatible formats.')

    # Sort the Filenames
    load_files = np.asarray(load_files)
    fidx = np.argsort(load_files)
    load_files = load_files[fidx]

    # Set Lat/Lon/Name for Current Station
    slat = my_lat
    slon = my_lon
    sname = my_sta

    # Determine the Chunk Sizes for the Convolution
    total_files = len(load_files)
    nominal_load = total_files // size # Floor Divide
    # Final Chunk Might Be Different in Size Than the Nominal Load
    if rank == size - 1:
        procN = total_files - rank * nominal_load
    else:
        procN = nominal_load

    # Perform the Convolution for Each Station
    if (rank == 0):
        eamp,epha,namp,npha,vamp,vpha = load_convolution.main(grn_file,norm_flag,load_files,regular,\
            slat,slon,sname,cnv_out,lsmask_file,lsmask_type,loadfile_format,rank,procN,comm,load_density=ldens)
    # For Worker Ranks, Run the Code But Don't Return Any Variables
    else:
        load_convolution.main(grn_file,norm_flag,load_files,regular,\
            slat,slon,sname,cnv_out,lsmask_file,lsmask_type,loadfile_format,rank,procN,comm,load_density=ldens)

    # Make Sure All Jobs Have Finished Before Continuing
    comm.Barrier()

# --------------------- END CODE --------------------------- #


