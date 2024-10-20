#!/usr/bin/env python

# *********************************************************************
# MAIN PROGRAM TO COMPUTE LOAD GREENS FUNCTIONS (DISPLACEMENT, GRAVITY, TILT, STRAIN)
#
# Copyright (c) 2014-2023: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
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

# IMPORT MPI MODULE
from mpi4py import MPI

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
import sys
import os
sys.path.append(os.getcwd() + "/../")

# IMPORT PYTHON MODULES
import numpy as np
from LOADGF.GF import compute_greens_functions

# --------------- SPECIFY USER INPUTS --------------------- #
 
# Full path to Load Love Number file (output from run_ln.py)
lln_file = ("../output/Love_Numbers/LLN/lln_PREM.txt")
#lln_file = ("../output/Love_Numbers/LLN/lln_Homogeneous_Vp05.92_Vs03.42_Rho03.00.txt")
#lln_file = ("../output/Love_Numbers/LLN/lln_Homogeneous_Vp05.92_Vs03.42_Rho03.00_nonGrav.txt")

# Output filename (Default is 'grn.txt')
file_out = ("PREM.txt")
#file_out = ("Homogeneous_Vp05.92_Vs03.42_Rho03.00.txt")
#file_out = ("Homogeneous_Vp05.92_Vs03.42_Rho03.00_nonGrav.txt")

# ------------------ END USER INPUTS ----------------------- #

# --------------------- SETUP MPI -------------------------- #

# Get the main MPI communicator that controls communication between processors
comm = MPI.COMM_WORLD
# Get my "rank", i.e. the processor number assigned to me
rank = comm.Get_rank()
# Get the total number of other processors used
size = comm.Get_size()

# ---------------------------------------------------------- #

# -------------------- BEGIN CODE -------------------------- #

# Ensure that the Output Directory Exists
if (rank == 0):
    if not (os.path.isdir("../output/Greens_Functions/")):
        os.makedirs("../output/Greens_Functions/")


# Make sure all jobs have finished before continuing
comm.Barrier()


# Compute the Displacement Greens functions (For Load Love Numbers Only)
if (rank == 0):
    u,v,u_norm,v_norm,u_cm,v_cm,u_norm_cm,v_norm_cm,u_cf,v_cf,u_norm_cf,v_norm_cf,gE,gE_norm,gE_cm,gE_cm_norm,\
        gE_cf,gE_cf_norm,tE,tE_norm,tE_cm,tE_cm_norm,tE_cf,tE_cf_norm,\
        e_tt,e_ll,e_rr,e_tt_norm,e_ll_norm,e_rr_norm,e_tt_cm,e_ll_cm,e_rr_cm,e_tt_cm_norm,e_ll_cm_norm,e_rr_cm_norm,\
        e_tt_cf,e_ll_cf,e_rr_cf,e_tt_cf_norm,e_ll_cf_norm,e_rr_cf_norm,gN,tN = \
            compute_greens_functions.main(lln_file,rank,comm,size,grn_out=file_out)
# For Worker Ranks, Run the Code But Don't Return Any Variables
else:
    compute_greens_functions.main(lln_file,rank,comm,size,grn_out=file_out)


# --------------------- END CODE --------------------------- #

