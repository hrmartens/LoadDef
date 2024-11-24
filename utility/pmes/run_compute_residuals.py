#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO COMPUTE RESIDUALS BETWEEN PARTICLE MOTION ELLIPSES (PMEs)
# LITERATURE: Martens et al. (2016, GJI)
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

# IMPORT ADDITIONAL MODULES
import numpy as np
from utility.pmes import compute_residuals

#### USER INPUTS ####

# Specify Inputs: Forward Models
harmonic = "M2"
model1 = "cm_convgf_TPXO9-Atlas_stationMesh_PREM"
model2 = "cm_convgf_FES2014_stationMesh_PREM"
input_directory = ("./output/")
filename1 = (input_directory + "pme_OceanOnly_" + harmonic + "_" + model1 + ".txt")
filename2 = (input_directory + "pme_OceanOnly_" + harmonic + "_" + model2 + ".txt")

# Optional: Select Individual Stations to Exclude When Computing the Common-Mode
stations_to_exclude = []

# Filter stations based on latitude?
filter_lat = False
keep_north = False # If False, keep stations to the south; if True, keep stations to the north
latitude = 50.

# Output filename
if filter_lat:
    if keep_north:
        outfile = ("Residuals_" + harmonic + "_" +  model1 + "-" + model2 + "_NorthStations.txt")
    else: # keep south
        outfile = ("Residuals_" + harmonic + "_" +  model1 + "-" + model2 + "_SouthStations.txt")
else:
    outfile = ("Residuals_" + harmonic + "_" +  model1 + "-" + model2 + ".txt")

#### BEGIN CODE

# Set-up Directory
if not (os.path.isdir("./output/Residuals/")):
    os.makedirs("./output/Residuals/")
outdir = "./output/Residuals/"

# Output File
myoutfile = (outdir + outfile)

# Filter on latitude?
if filter_lat:
    stations = np.loadtxt(filename1,skiprows=1,unpack=True,usecols=(0,),dtype='U')
    slat = np.loadtxt(filename1,skiprows=1,unpack=True,usecols=(1,))
    if keep_north:
        idx_to_exclude = np.where(slat < latitude)[0]
    else:
        idx_to_exclude = np.where(slat >= latitude)[0]
    stations_to_exclude.extend(stations[idx_to_exclude].tolist())

# Compute Residuals and Remove Common Mode
rmCMode = True
compute_residuals.main(filename1,filename2,myoutfile,rmCMode,stations_to_exclude)

# Compute Residuals but do not Remove Common Mode
rmCMode = False
compute_residuals.main(filename1,filename2,myoutfile,rmCMode,stations_to_exclude)


