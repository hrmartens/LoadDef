# *********************************************************************
# FUNCTION TO READ IN THE GOT4.10 OCEAN TIDE MODELS
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
from math import pi
import sys
import os

def main(filename):

    # Create Temp Files
    temp_amp = ("./temp_amp"+str(np.random.randint(500))+".txt")
    temp_pha = ("./temp_pha"+str(np.random.randint(500))+".txt")

    # Open Full Ascii File, Read Line-by-Line, and Partition into Separate Amp & Pha Files
    outfile_amp = open(temp_amp, 'w')
    outfile_pha = open(temp_pha, 'w')
    with open(filename, 'r') as myfile:
        for line in myfile:
            # Search for the amplitude section
            if "amplitude" in line:
                flag = 1
                header = 0
            # Search for the phase section (note the capital P)
            if "phase" in line:
                flag = 2
                header = 0 # reset the header counter
            if (flag == 1): 
                header += 1
                if (header == 3):
                    latdim,londim = line.split()
                    latdim = int(latdim)
                    londim = int(londim)
                if (header == 4):
                    minlat,maxlat = line.split()
                    minlat = float(minlat)
                    maxlat = float(maxlat)
                if (header == 5):
                    minlon,maxlon = line.split()
                    minlon = float(minlon)
                    maxlon = float(maxlon)
                if (header == 6): 
                    fillval,fillval = line.split()
                    fillval = float(fillval)
                if (header > 7): # Data begins after 7 headerlines
                    outfile_amp.write(line)
            if (flag == 2):
                header += 1
                if (header > 7):
                    outfile_pha.write(line)
    outfile_amp.close()
    outfile_pha.close()

    # Read Amplitude File
    with open(temp_amp) as f_amp:
        amp = np.array(f_amp.read().split(), dtype=float) #.reshape(latdim,londim)
    # Read Phase File
    with open(temp_pha) as f_pha:
        pha = np.array(f_pha.read().split(), dtype=float) #.reshape(latdim,londim)
    
    # Determine Lat/Lon Values
    lat1dseq = np.linspace(minlat,maxlat,latdim)
    lon1dseq = np.linspace(minlon,maxlon,londim)

    # Set Fill Values to Zero
    amp[amp == fillval] = 0.
    pha[pha == fillval] = 0.

    # Convert Amplitude from Centimeters to Meters
    amp  = np.divide(amp,100.)

    # Write Phase to New Array to Make it Writeable 
    pha  = np.divide(pha,1.) 
 
    # Define Full Lon/Lat/Amp/Pha Arrays
    grid_olon, grid_olat = sc.meshgrid(lon1dseq,lat1dseq)
    olon = grid_olon.flatten()
    olat = grid_olat.flatten()
    amp2darr = np.reshape(amp,(latdim,londim))
    pha2darr = np.reshape(pha,(latdim,londim))

    # Remove Temp Files
    os.remove(temp_amp)
    os.remove(temp_pha)

    # Return Parameters
    return olat,olon,amp,pha,lat1dseq,lon1dseq,amp2darr,pha2darr

