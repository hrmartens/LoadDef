#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO COMBINE STATION FILES FOR DIFFERENT LOADS INTO A 
#   SINGLE FILE FOR THE NETWORK (ONE FILE PER LOAD MODEL)
#
# LITERATURE: Martens et al. (2016, GJI), Martens (2016, Caltech)
# 
# Copyright (c) 2014-2024: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
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

# Import Python Modules
from CONVGF.utility import read_convolution_file
import numpy as np

#### USER INPUT ####
directory = ("../../output/Convolution/")
prefix = ("cn_LandAndOceans_")
pmod = ("PREM")
#pmod = ("Homogeneous_Vp05.92_Vs03.42_Rho03.00")
#pmod = ("Homogeneous_Vp05.92_Vs03.42_Rho03.00_nonGrav")
suffix = ("ce_convgf_custom_symcaps_" + pmod + "_z111_z215_z320_z430_azm0p5.txt")
#suffix = ("ce_convgf_custom_symcaps_" + pmod + "_commonMesh_0.001-11.0_0.005-15.0_0.01-20.0_0.1-30.0_0.5-150.0_0.1-160.0_0.01-165.0_0.005-169.0_0.001.txt")
#suffix = ("ce_convgf_custom_symcaps_" + pmod + "_commonMesh_0.5_0.5_60.0_90.0_0.1_0.5_70.0_90.0_0.01_0.5_75.0_90.0_0.005_0.5_79.0_90.0_0.001_0.5.txt")

#### BEGIN CODE ####

# Create output directory, if it does not yet exist
if not (os.path.isdir("./output/")):
    os.makedirs("./output/")
output_directory = ("./output/")

# Locate applicable files
station_files = []
for myfile in os.listdir(directory):
    if myfile.endswith(suffix):
        if myfile.startswith(prefix):
            station_files.append(myfile)

# Check for files
if not station_files:
    sys.exit('No station files match the criteria.')

# Sort the Filenames
station_files = np.sort(station_files)

# Loop through files
for ii in range(0,len(station_files)):
 
    # Current file
    cfile = station_files[ii]
    print('Working on file: %s'%cfile)

    # Extract station name
    try:
        start = cfile.index(prefix) + len(prefix)
        end = cfile.index(suffix,start)
        cstation = cfile[start:end]
    except ValueError:
        print("Error: Could not find station name.")

    # Read the current file
    extension,lat,lon,eamp,epha,namp,npha,vamp,vpha = read_convolution_file.main(directory + cfile)

    # Number of elements
    if extension.shape:
        numel = len(extension)
    else:
        numel = 1

    # Extract info about each harmonic
    for jj in range(0,numel):

        # Multiple rows of data
        if (numel > 1):
            cext = extension[jj]
            ceamp = eamp[jj]
            cepha = epha[jj]
            cnamp = namp[jj]
            cnpha = npha[jj]
            cvamp = vamp[jj]
            cvpha = vpha[jj]
            clat = lat[jj]
            clon = lon[jj]
        # Only one row of data
        else:
            cext = str(extension)
            ceamp = eamp
            cepha = epha
            cnamp = namp
            cnpha = npha
            cvamp = vamp
            cvpha = vpha
            clat = lat
            clon = lon

        # Prepare output files
        cnv_file = (output_directory + prefix + cext + "_" + suffix)
        cnv_filename = (prefix + cext + suffix)
        cnv_head = (output_directory + prefix + str(np.random.randint(500))+ "head.txt")
        cnv_body = (output_directory + prefix + str(np.random.randint(500))+ "body.txt")

        # Convert floats to string [recently added]
        clat = "{:.8f}".format(clat)
        clon = "{:.8f}".format(clon)
        ceamp = "{:.8f}".format(ceamp)
        cepha = "{:.8f}".format(cepha)
        cnamp = "{:.8f}".format(cnamp)
        cnpha = "{:.8f}".format(cnpha)
        cvamp = "{:.8f}".format(cvamp)
        cvpha = "{:.8f}".format(cvpha)

        # Prepare data for output 
        all_cnv_data = [cstation, clat, clon, ceamp, cepha, cnamp, cnpha, cvamp, cvpha]
        all_cnv_data = np.asarray(all_cnv_data)

        # Delete old files
        if (ii == 0):
            if os.path.isfile(cnv_file):
                print('Removing old file: %s' %cnv_file)
                os.remove(cnv_file)

        # Test for existence of file
        if not os.path.isfile(cnv_file):

            # Write Header Info to File
            hf = open(cnv_head,'w')
            cnv_str = 'Station  Lat(+N,deg)  Lon(+E,deg)  E-Amp(mm)  E-Pha(deg)  N-Amp(mm)  N-Pha(deg)  V-Amp(mm)  V-Pha(deg) \n'
            hf.write(cnv_str)
            hf.close()
 
            # Write Convolution Results to File
            np.savetxt(cnv_body,all_cnv_data.reshape(1, all_cnv_data.shape[0]),fmt="%s",delimiter="      ")
     
            # Combine Header and Body Files
            filenames_cnv = [cnv_head, cnv_body]
            with open(cnv_file,'w') as outfile:
                for fname in filenames_cnv:
                    with open(fname) as infile:
                        outfile.write(infile.read())

            # Remove Header and Body Files
            os.remove(cnv_head)
            os.remove(cnv_body)

        # Append to existing file
        else:

            # Write Convolution Results to File
            np.savetxt(cnv_body,all_cnv_data.reshape(1, all_cnv_data.shape[0]),fmt="%s",delimiter="      ")

            # Combine Existing and Body Files
            with open(cnv_file,'a') as outfile:
                with open(cnv_body) as infile:
                    outfile.write(infile.read())

            # Remove Header and Body Files
            os.remove(cnv_body)

