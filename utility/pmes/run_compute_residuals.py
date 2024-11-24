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
from utility.pmes import compute_residuals

#### USER INPUTS ####

# Optional Stations to Exclude When Computing the Common-Mode
stations_to_exclude = []

# Specify Inputs: Forward Models
harmonic = "M2"
rfm = "cm"
model1 = "FES2014"
model2 = "GOT410c"
suffix = "commonMesh_PREM"
input_directory = ("./output/")
filename1 = (input_directory + "pme_OceanOnly_" + harmonic + "_" + rfm + "_convgf_" + model1 + "_" + suffix + ".txt")
filename2 = (input_directory + "pme_OceanOnly_" + harmonic + "_" + rfm + "_convgf_" + model2 + "_" + suffix + ".txt")
outfile = ("Residuals_" + harmonic + "_" + rfm + "_" + model1 + "-" + model2 + "_" + suffix + ".txt")

#### BEGIN CODE

# Set-up Directory
if not (os.path.isdir("./output/Residuals/")):
    os.makedirs("./output/Residuals/")
outdir = "./output/Residuals/"

# Output File
myoutfile = (outdir + outfile)

# Compute Residuals and Remove Common Mode
rmCMode = True
compute_residuals.main(filename1,filename2,myoutfile,rmCMode,stations_to_exclude)

# Compute Residuals but do not Remove Common Mode
rmCMode = False
compute_residuals.main(filename1,filename2,myoutfile,rmCMode,stations_to_exclude)


