#!/usr/bin/env python

# *********************************************************************
# MAIN PROGRAM TO COMPUTE LOVE NUMBER PARTIAL DERIVATIVES
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
from LOADGF.PL import compute_ln_partials

# --------------- SPECIFY USER INPUTS --------------------- #

# Full path to planet model text file
# :: Planet model should be spherically symmetric, elastic, 
#    non-rotating, and isotropic (SNREI)
# :: Format: radius(km), vp(km/s), vs(km/s), density(g/cc)
# :: If the file delimiter is not whitespace, then specify in
#    call to function. 
planet_model = ("../input/Planet_Models/PREM.txt")

# Extension for the output filename (Default is '.txt')
file_ext      = ("PREM_partials.txt")

# Specify range of spherical harmonic degrees to be computed for the Love number partial derivatives
firstn       = 0
finaln       = 4

# Reproduce Figure 1 from Okubo & Endo (1986) as a sanity check
plot_fig = False
 
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
    if not (os.path.isdir("../output/Love_Numbers/Partials/")):
        os.makedirs("../output/Love_Numbers/Partials")


# Make sure all jobs have finished before continuing
comm.Barrier()


# Compute the Love numbers (Load and Potential)
if (rank == 0):

    ln_n,ln_h,ln_nl,ln_nk,ln_h_inf,ln_l_inf,ln_k_inf,ln_h_inf_p,ln_l_inf_p,ln_k_inf_p,\
        ln_hpot,ln_nlpot,ln_nkpot,ln_hstr,ln_nlstr,ln_nkstr,ln_hshr,ln_nlshr,ln_nkshr,\
        ln_planet_radius,ln_planet_mass,ln_sint,ln_Yload,ln_Ypot,ln_Ystr,ln_Yshr,\
        ln_lmda_surface,ln_mu_surface = \
        compute_love_numbers.main(planet_model,rank,comm,size,file_out=file_ext,startn=firstn,stopn=finaln)

# For Worker Ranks, Run the Code But Don't Return Any Variables
else:

    compute_love_numbers.main(planet_model,rank,comm,size,file_out=file_ext,startn=firstn,stopn=finaln)

    # Workers Will Know Nothing About the Data Used to Compute the GFs
    ln_n = ln_h = ln_nl = ln_nk = ln_h_inf = ln_l_inf = ln_k_inf = ln_h_inf_p = ln_l_inf_p = ln_k_inf_p = None
    ln_planet_radius = ln_planet_mass = ln_Yload = ln_Ypot = ln_Ystr = ln_Yshr = None
    ln_hpot = ln_nlpot = ln_nkpot = ln_hstr = ln_nlstr = ln_nkstr = ln_hshr = None
    ln_nlshr = ln_nkshr = ln_sint = ln_lmda_surface = ln_mu_surface = None


# Make sure all jobs have finished before continuing
comm.Barrier()


# Broadcast Arguments to All Ranks in Preparation for Partials Computation
ln_n             = comm.bcast(ln_n, root=0)
ln_h             = comm.bcast(ln_h, root=0)
ln_nl            = comm.bcast(ln_nl, root=0)
ln_nk            = comm.bcast(ln_nk, root=0)
ln_hpot          = comm.bcast(ln_hpot, root=0)
ln_nlpot         = comm.bcast(ln_nlpot, root=0)
ln_nkpot         = comm.bcast(ln_nkpot, root=0)
ln_hshr          = comm.bcast(ln_hshr, root=0)
ln_nlshr         = comm.bcast(ln_nlshr, root=0)
ln_nkshr         = comm.bcast(ln_nkshr, root=0)
ln_hstr          = comm.bcast(ln_hstr, root=0)
ln_nlstr         = comm.bcast(ln_nlstr, root=0)
ln_nkstr         = comm.bcast(ln_nkstr, root=0)
ln_planet_radius  = comm.bcast(ln_planet_radius, root=0)
ln_planet_mass    = comm.bcast(ln_planet_mass, root=0)
ln_sint          = comm.bcast(ln_sint, root=0)
ln_Yload         = comm.bcast(ln_Yload, root=0)
ln_Ypot          = comm.bcast(ln_Ypot, root=0)
ln_Ystr          = comm.bcast(ln_Ystr, root=0)
ln_Yshr          = comm.bcast(ln_Yshr, root=0)


# Compute Partial Derivatives
if (rank == 0):

    dht_dmu,dlt_dmu,dkt_dmu,dht_dK,dlt_dK,dkt_dK,dht_drho,dlt_drho,dkt_drho,dhl_dmu,dll_dmu,dkl_dmu,dhl_dK,dll_dK,dkl_dK,\
        dhl_drho,dll_drho,dkl_drho,dhs_dmu,dls_dmu,dks_dmu,dhs_dK,dls_dK,dks_dK,dhs_drho,dls_drho,dks_drho = \
        compute_ln_partials.main(ln_n,ln_sint,ln_Yload,ln_Ypot,ln_Yshr,ln_Ystr,ln_h,ln_nl,ln_nk,ln_hpot,ln_nlpot,ln_nkpot,ln_hshr,ln_nlshr,ln_nkshr,\
        ln_hstr,ln_nlstr,ln_nkstr,planet_model,rank,comm,size,par_out=file_ext,plot_figure=plot_fig)

# For Worker Ranks, Run the Code But Don't Return Any Variables
else:

    compute_ln_partials.main(ln_n,ln_sint,ln_Yload,ln_Ypot,ln_Yshr,ln_Ystr,ln_h,ln_nl,ln_nk,ln_hpot,ln_nlpot,ln_nkpot,ln_hshr,ln_nlshr,ln_nkshr,\
        ln_hstr,ln_nlstr,ln_nkstr,planet_model,rank,comm,size,par_out=file_ext,plot_figure=plot_fig)

    # Workers Will Know Nothing About the Data
    dht_dmu = dlt_dmu = dkt_dmu = dht_dK = dlt_dK = dkt_dK = dht_drho = dlt_drho = dkt_drho = None
    dhl_dmu = dll_dmu = dkl_dmu = dhl_dK = dll_dK = dkl_dK = dhl_drho = dll_drho = dkl_drho = None
    dhs_dmu = dls_dmu = dks_dmu = dhs_dK = dls_dK = dks_dK = dhs_drho = dls_drho = dks_drho = None

# --------------------- END CODE --------------------------- #

