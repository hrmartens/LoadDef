#!/usr/bin/env python

# *********************************************************************
# MAIN PROGRAM TO COMPUTE LOVE NUMBERS (POTENTIAL/TIDE, LOAD, SHEAR, STRESS)
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

# *********************************************************************
# Special note: We now include a new keyword agrument in the call to 
#               compute_love_numbers ("nongrav"), which toggles self-
#               gravity on or off. Please note that the computation 
#               without self-gravity (nongrav=True) is not yet well 
#               tested. Proceed with caution and check your results. 
#               The default is to include self-gravity (nongrav=False).
#
# Special note: If desiring to compute Love numbers at depths interior
#               to the planet (and/or at multiple depths), then you can
#               now set a new keyword parameter, "eval_radii", to a list
#               of the radii (in meters) at which you want to evaluate
#               the Love numbers. 
#               Example: eval_radii = [6356000,6331000,6167000,6371000]
#               If computing Love numbers at a radius other than the
#               surface, it is recommended to increase the value of
#               "num_soln" (also a keyword argument) to help with 
#               honing in on the correct radius.
#               Example: num_soln = 500
#               As a new option, eval_radii is relatively untested.
#               As always, proceed with caution and check your results.
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
 
# Extension for the output filename (Default is '.txt')
file_ext      = ("PREM.txt")

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
        compute_love_numbers.main(planet_model,rank,comm,size,file_out=file_ext)
# For Worker Ranks, Run the Code But Don't Return Any Variables
else: 
    # Workers Compute Love Numbers
    compute_love_numbers.main(planet_model,rank,comm,size,file_out=file_ext)
    # Workers Will Know Nothing About the Data Used to Compute the GFs
    ln_n = ln_h = ln_nl = ln_nk = ln_h_inf = ln_l_inf = ln_k_inf = ln_h_inf_p = ln_l_inf_p = ln_k_inf_p = None
    ln_planet_radius = ln_planet_mass = ln_Yload = ln_Ypot = ln_Ystr = ln_Yshr = None
    ln_hpot = ln_nlpot = ln_nkpot = ln_hstr = ln_nlstr = ln_nkstr = ln_hshr = None
    ln_nlshr = ln_nkshr = ln_sint = ln_lmda_surface = ln_mu_surface = None 

# --------------------- END CODE --------------------------- #

