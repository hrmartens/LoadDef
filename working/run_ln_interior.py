#!/usr/bin/env python

# *********************************************************************
# MAIN PROGRAM TO COMPUTE LOVE NUMBERS (POTENTIAL/TIDE, LOAD, SHEAR, STRESS)
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

# IMPORT MPI MODULE
from mpi4py import MPI

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
import sys
import os
sys.path.append(os.getcwd() + "/../")

# IMPORT PYTHON MODULES
import numpy as np
from LOADGF.LN import compute_love_numbers 

# --------------- SPECIFY USER INPUTS --------------------- #
 
# Full path to planet model text file
#     Planet model should be spherically symmetric, elastic, 
#         non-rotating, and isotropic (SNREI)
#     Format: radius(km), vp(km/s), vs(km/s), density(g/cc)
#     If the file delimiter is not whitespace, then specify in
#         call to function. 
planet_model = ("../input/Planet_Models/PREM.txt")

# Radius at which to evaluate the Love numbers (meters)
radius_for_evaluation = 6356000
num_soln = 10000 # helps to hone in on the correct radius
 
# Extension for the output filename (Default is '.txt')
file_ext      = ("PREM_" + str(radius_for_evaluation) + ".txt")

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

# Ensure that the Output Directories Exist
if (rank == 0):
    if not (os.path.isdir("../output/Love_Numbers/")):
        os.makedirs("../output/Love_Numbers/")
    if not (os.path.isdir("../output/Love_Numbers/LLN/")):
        os.makedirs("../output/Love_Numbers/LLN")
    if not (os.path.isdir("../output/Love_Numbers/PLN/")):
        os.makedirs("../output/Love_Numbers/PLN") 
    if not (os.path.isdir("../output/Love_Numbers/STR/")):
        os.makedirs("../output/Love_Numbers/STR")   
    if not (os.path.isdir("../output/Love_Numbers/SHR/")):
        os.makedirs("../output/Love_Numbers/SHR")


# Make sure all jobs have finished before continuing
comm.Barrier()


# Compute the Love numbers (Load and Potential)
if (rank == 0):
    # Compute Love Numbers
    ln_n,ln_h,ln_nl,ln_nk,ln_h_inf,ln_l_inf,ln_k_inf,ln_h_inf_p,ln_l_inf_p,ln_k_inf_p,\
        ln_hpot,ln_nlpot,ln_nkpot,ln_hstr,ln_nlstr,ln_nkstr,ln_hshr,ln_nlshr,ln_nkshr,\
        ln_planet_radius,ln_planet_mass,ln_sint,ln_Yload,ln_Ypot,ln_Ystr,ln_Yshr,\
        ln_lmda_surface,ln_mu_surface = \
        compute_love_numbers.main(planet_model,rank,comm,size,file_out=file_ext,eval_radius=radius_for_evaluation,num_soln=num_soln)#,interp_emod=True)
# For Worker Ranks, Run the Code But Don't Return Any Variables
else: 
    # Workers Compute Love Numbers
    compute_love_numbers.main(planet_model,rank,comm,size,file_out=file_ext,eval_radius=radius_for_evaluation,num_soln=num_soln)#,interp_emod=True)
    # Workers Will Know Nothing About the Data Used to Compute the GFs
    ln_n = ln_h = ln_nl = ln_nk = ln_h_inf = ln_l_inf = ln_k_inf = ln_h_inf_p = ln_l_inf_p = ln_k_inf_p = None
    ln_planet_radius = ln_planet_mass = ln_Yload = ln_Ypot = ln_Ystr = ln_Yshr = None
    ln_hpot = ln_nlpot = ln_nkpot = ln_hstr = ln_nlstr = ln_nkstr = ln_hshr = None
    ln_nlshr = ln_nkshr = ln_sint = ln_lmda_surface = ln_mu_surface = None 

# --------------------- END CODE --------------------------- #

