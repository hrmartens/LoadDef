#!/usr/bin/env python

# *********************************************************************
# MAIN PROGRAM TO INVERT OBSERVED SURFACE DISPLACEMENTS FOR STRUCTURE
# 
# Copyright (c) 2022-2024 | Hilary R. Martens
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

# MODIFY PYTHON PATH TO INCLUDE 'StructSolv' DIRECTORY
import sys
import os
sys.path.append(os.getcwd() + "/../../")

# IMPORT PYTHON MODULES
import numpy as np
import scipy as sc
import datetime
import netCDF4 
from structinv.solv import structure_inversion
from structinv.utility import read_structureDM

# --------------- SPECIFY USER INPUTS --------------------- #
 
### -- DESIGN MATRIX -- (created with run_dm_structure.py in LoadDef)
dm_file = ("../output/DesignMatrixStructure/designmatrix_cn_OceanOnly_GOT410c-M2_cm_convgf_GOT410c_PREM_dens1030_commonMesh.nc")

### Have both real and imaginary components been included in the design matrix? Set "inc_imag" = True if imaginary components ARE included. 
inc_imag = True

### -- STARTING MODEL -- Gm0, where m0 is the starting model vector (e.g., PREM). (can also be created with run_dm_structure.py in LoadDef)
##  NOTE: One header line is assumed.
##  Two options for formats depending on whether (1) only real or (2) both real and imaginary components are included in the design matrix:
##   OPTION 1 (Only Real): Station,  Lat(+N,deg),  Lon(+E,deg),  E-Disp(mm),  N-Disp(mm),  U-Disp(mm)
##   OPTION 2 (Harmonic): Station, Lat(+N,deg), Lon(+E,deg), E-Disp-Re(mm), N-Disp-Re(mm), U-Disp-Re(mm), E-Disp-Im(mm), N-Disp-Im(mm), U-Disp-Im(mm)
##   The "inc_imag" parameter will determine which format is read.
startmod = ("../output/DesignMatrixStructure/startingmodel_cn_OceanOnly_GOT410c-M2_cm_convgf_GOT410c_PREM_dens1030_commonMesh.txt")
 
### -- DATA -- 
##  NOTE: One header line is assumed.
##  Two options for formats depending on whether (1) only real or (2) both real and imaginary components are included in the design matrix:
##   OPTION 1 (Only Real): Station, Latitude[+N], Longitude[+E], East-Displacement[mm], North-Displacement[mm], Up-Displacement[mm]
##   OPTION 2 (Harmonic): Station, Latitude[+N], Longitude[+E], East-Amp[mm], East-Pha[deg], North-Amp[mm], North-Pha[deg], Up-Amp[mm], Up-Pha[deg]
##   The "inc_imag" parameter will determine which format is read.
##   Note: If only up-component data are available, please set "uonly" flag below to True. 
data_dir = ("../input/GNSS-Data/") 
##   Datafile prefix
data_pre = ("M2_ObsEllipses")
##  Does the datafile contain up-component data ONLY? If so, set uonly = True. Default is uonly = False.
##   OPTION 1 (Only Real): Station, Latitude[+N], Longitude[+E], Up-Displacement[mm]
##   OPTION 2 (Harmonic): Station, Latitude[+N], Longitude[+E], Up-Amp[mm], Up-Pha[deg]
##   The "inc_imag" parameter will determine which format is read.
##   Note: It is always assumed that the Design Matrix is built based on three components; this parameter applies only to the data!]
uonly = False
##  Does the datafile contain information about particle-motion ellipses? If so, set "pme = True." 
##   Note: It is assumed that the P.M.E. information is in columns 4,5,6 (starting from 1), which is the output of PyTide.
pme = True
 
### Tikhonov regularization parameter and order 
### Options for order: 'zeroth', 'second', 'zeroth_second', and 'none')
### "zeroth_second" applies both zeroth and second order regularization
alpha = 1 # second order
beta = 1 # zeroth order
tikhonov = 'second'

### Output file suffix
outfile = ("_TikReg-" + tikhonov + "-" + str(alpha) + ".txt")

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
    if not (os.path.isdir("../output/Inversion/")):
        os.makedirs("../output/Inversion/")
    if not (os.path.isdir("../output/Inversion/Structure/")):
        os.makedirs("../output/Inversion/Structure/")

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
    model_vector = structure_inversion.main(dm_file,data_files,startmod,rank,procN,comm,alpha=alpha,beta=beta,tikhonov=tikhonov,outfile=outfile,uonly=uonly,imaginary=inc_imag,pme=pme)
# Worker Ranks
else:
    structure_inversion.main(dm_file,data_files,startmod,rank,procN,comm,alpha=alpha,tikhonov=tikhonov,outfile=outfile,uonly=uonly,imaginary=inc_imag,pme=pme)

# Print the model vector
print(':: ')
print(':: ')
print(':: Model Solution Vector: ')
print(model_vector)

# Read the design matrix to understand the model vector
design_matrix,sta_comp_ids,sta_comp_lat,sta_comp_lon,perturb_radius_bottom,perturb_radius_top,perturb_param = read_structureDM.main(dm_file)
print(':: Bottom radii of perturbed layers: ')
print(perturb_radius_bottom)
print(':: Top radii of perturbed layers: ')
print(perturb_radius_top)
print(':: Material parameter perturbed: ')
print(perturb_param) 

# --------------------- END CODE --------------------------- #


