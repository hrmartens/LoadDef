#!/usr/bin/env python

# *********************************************************************
# MAIN PROGRAM TO INVERT OBSERVED SURFACE DISPLACEMENTS FOR SURFACE-
#  LOAD DISTRIBUTION
#
# Copyright (c) 2021-2024 | Hilary R. Martens
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
sys.path.append(os.getcwd() + "/../../")

# IMPORT PYTHON MODULES
import numpy as np
import scipy as sc
import datetime
import netCDF4 
from loadinv.solv import load_inversion
from loadinv.utility import read_loadDesignMatrix

# --------------- SPECIFY USER INPUTS --------------------- #

### Design matrix file (can be created using "LoadDef" software)
dm_file = ("../input/DesignMatrix/designmatrix_cf_PREM_cells_28.0_50.0_233.0_258.0_0.25_commonMesh_regional_28.0_50.0_233.0_258.0_0.01_0.01_oceanmask_NOTA_Select.nc")
 
### Data directory 
###  Datafile Format: Station, Latitude[+N], Longitude[+E], East-Displacement[mm], North-Displacement[mm], Up-Displacement[mm]
###      If only vertical-component data are available, please set "uonly" flag below to True. 
data_dir = ("../input/GNSS-Data/")

### Datafile prefix
data_pre = ("NOTA_fake")

###  Does the datafile contain only vertical-component data? If so, set uonly = True. Default is uonly = False.
###      Datafile Format: Station, Latitude[+N], Longitude[+E], Up-Displacement[mm]
###      [NOTE: It is always assumed that the design matrix is built based on three components; this parameter applies only to the data!]
uonly = True
 
### Tikhonov regularization parameter and order.
### Options: 'zeroth', 'second', 'zeroth_second', 'none')
alpha = 0.1
beta = 0.1
tikhonov = 'zeroth_second'

### Reference Height (m) and Density (kg/m^3) of Load Used to Compute the Design Matrix 
ref_height = 1.
ref_density = 1000.

### Output file suffix
outfile = ("_TikReg-" + tikhonov + "-alpha" + str(alpha) + "-beta" + str(beta) + ".txt")

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
    if not (os.path.isdir("../output/")):
        os.makedirs("../output/")
    if not (os.path.isdir("../output/Inversion/")):
        os.makedirs("../output/Inversion/")
    if not (os.path.isdir("../output/Inversion/SurfaceLoad/")):
        os.makedirs("../output/Inversion/SurfaceLoad/")

# Determine Number of Available Data Files
data_files = []
if os.path.isdir(data_dir):
    for mfile in os.listdir(data_dir): # Filter by Data Directory
        if mfile.startswith(data_pre): # Filter by Datafile Prefix
            data_files.append(data_dir + mfile) # Append File to List
else:
    sys.exit('Error: Data directory not found.')

# Test for Data Files
if not data_files:
    sys.exit('Error: Data files not found.')

# Determine Number of Files Read In
if isinstance(data_files,float) == True: # only 1 file
    numel = 1 
else:
    numel = len(data_files)

# Determine the Chunk Sizes for the Inversion
total_files = len(data_files)
nominal_load = total_files // size # Floor Divide
# Final Chunk Might Be Different in Size Than the Nominal Load
if rank == size - 1:
    procN = total_files - rank * nominal_load
else:
    procN = nominal_load

# Perform the Inversion(s) 
# Primary Rank
if (rank == 0):
    model_vector = load_inversion.main(dm_file,data_files,rank,procN,comm,reference_height=ref_height,reference_density=ref_density,tikhonov=tikhonov,alpha=alpha,beta=beta,outfile=outfile,uonly=uonly)
# Worker Ranks
else:
    load_inversion.main(dm_file,data_files,rank,procN,comm,reference_height=ref_height,reference_density=ref_density,tikhonov=tikhonov,alpha=alpha,beta=beta,outfile=outfile,uonly=uonly)

# --------------------- END CODE --------------------------- #


