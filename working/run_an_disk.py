#!/usr/bin/env python

# *********************************************************************
# SPECIAL PROGRAM TO COMPUTE THE VERTICAL AND HORIZONTAL DISPLACEMENT 
#  RESPONSE TO A DISK LOAD USING AN ANALYTICAL APPROACH BASED ON THE 
#  DISK FACTOR FROM FARRELL (1972); NO CONVOLUTION NECESSARY.
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
# SPECIAL NOTE: This program will compute the displacement response to 
#               a disk load without the need to run a separate convolution. 
#               We take advantage of the simple disk geometry and the disk
#               factor described in Farrell (1972) to compute the 
#               deformation response to a disk load of arbitrary radius,
#               height, and density (user specified). 
#
#               Note that the displacement results will appear in the 
#               Green's-function file itself (columns 2 and 3). 
#               Angular distance from the center of the disk appears in 
#               the first column. Ignore all other columns (4+).
#               In other words, the disp LGFs are no longer for a point load, 
#               but for the finite-sized disk load specified by the user. 
#
#               If investigating the response to loading at both poles, 
#               simply add the Green's functions together,  
#               after reversing the angular order of one set. 
#
#               This is a relatively new addition to LoadDef and is 
#               therefore not as extensively tested as other features.
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
from LOADGF.GF import compute_greens_functions

# --------------- SPECIFY USER INPUTS --------------------- #
 
# Full path to Load Love Number file (output from run_ln.py)
lln_file = ("../output/Love_Numbers/LLN/lln_PREM.txt")
#lln_file = ("../output/Love_Numbers/LLN/lln_Homogeneous_Vp05.92_Vs03.42_Rho03.00.txt")
 
# Output filename (Default is 'grn.txt')
file_out = ("PREM_analyticalDisk.txt")
#file_out = ("Homogeneous_Vp05.92_Vs03.42_Rho03.00_analyticalDisk.txt")

# Apply a disk factor everywhere, with a radius of 'disks' degrees
diskf = True # True tells LoadDef to apply the disk factor
angdst = 0. # The angular distance from the load point at which to start applying the disk factor (for analytical disk, angdst must be set to 0.)
disks = 10. # The angular radius of the disk (in degrees)

#### Maximum theta value beyond which asymptotic approximations -- i.e. Kummer's transformation -- are not used.
#### NOTE: MUST set max_theta = 0 when computing analytical displacement response to a disk load.
#### This avoids the Kummer's transformation (the disk factor is not incorporated in the asymptotic series).
max_theta = 0.

# We need to compute the area of the disk (on a sphere)
#   For the analytical approach, we must integrate analytically over the surface of the sphere 
#    (this is in lieu of the convolution)
# Note: Int_phi=0^2pi Int_theta=0^disks r^2 sin(theta) dphi dtheta
#   computes the area of the disk at the pole
# Disk area for a unit-radius sphere = 2 * pi * -(cos(disk_radius) - cos(0)) = 2 * pi * (1 - cos(disk_radius))
# USER INPUTS:
planet_radius = 6371000. # units of meters
height_of_load = 1. # units of meters
density_of_load = 1000. # units of [kg/m^3]
# COMPUTATIONS BASED ON USER INPUTS:
disk_area_unit_sphere = 2. * np.pi * -(np.cos(np.radians(disks)) - np.cos(np.radians(0)))
disk_area = (planet_radius)**2 * disk_area_unit_sphere
volume_of_load = disk_area * height_of_load
mass_of_load = volume_of_load * density_of_load # units of kg

# Specify theta values
theta = [0.001,0.01,0.1,0.2,0.4,0.6,0.8,1.0,2.,3.,4.,5.,6.,7.,8.,9.,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,10., \
            10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23., \
            24.,25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.,37.,38.,39.,40.,41.,42.,43.,44.,45.,46.,47.,48.,49.,50., \
            51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,69.,70.,71.,72.,73.,74.,75.,76.,77., \
            78.,79.,80.,81.,82.,83.,84.,85.,86.,87.,88.,89.,90., \
            91.,92.,93.,94.,95.,96.,97.,98.,99.,100.,101.,102.,103.,104.,105.,106.,107.,108.,109.,110.,111.,112.,113.,114., \
            115.,116.,117.,118.,119.,120.,121.,122.,123.,124.,125.,126.,127.,128.,129.,130.,131.,132.,133.,134.,135.,136., \
            137.,138.,139.,140.,141.,142.,143.,144.,145.,146.,147.,148.,149.,150.,151.,152.,153.,154.,155.,156.,157.,158., \
            159.,160.,161.,162.,163.,164.,165.,166.,167.,168.,169.,169.1,169.2,169.3,169.4,169.5,169.6,169.7,169.8,169.9,170., \
            170.1,170.2,170.3,170.4,170.5,170.6,170.7,170.8,170.9,171.,172.,173.,174.,175.,176.,177.,178.,179., \
            179.2,179.4,179.6,179.8,179.9,179.99,179.999]

# NOTE: Plot results using ../utility/plots/plot_ad.py

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
            compute_greens_functions.main(lln_file,rank,comm,size,grn_out=file_out,disk_factor=diskf,angdist=angdst,disk_size=disks,loadmass=mass_of_load,max_theta=max_theta,theta=theta)
# For Worker Ranks, Run the Code But Don't Return Any Variables
else:
    compute_greens_functions.main(lln_file,rank,comm,size,grn_out=file_out,disk_factor=diskf,angdist=angdst,disk_size=disks,loadmass=mass_of_load,max_theta=max_theta,theta=theta)

# --------------------- END CODE --------------------------- #

# Specify theta values
#theta = [0.001,0.01,0.1,0.2,0.4,0.6,0.8,1.0,2.,3.,4.,5.,6.,7.,8.,9.,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,10., \
#            10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23., \
#            24.,25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.,37.,38.,39.,40.,41.,42.,43.,44.,45.,46.,47.,48.,49.,50., \
#            51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,69.,70.,71.,72.,73.,74.,75.,76.,77., \
#            78.,79.,80.,81.,82.,83.,84.,85.,86.,87.,88.,89.,90., \
#            91.,92.,93.,94.,95.,96.,97.,98.,99.,100.,101.,102.,103.,104.,105.,106.,107.,108.,109.,110.,111.,112.,113.,114., \
#            115.,116.,117.,118.,119.,120.,121.,122.,123.,124.,125.,126.,127.,128.,129.,130.,131.,132.,133.,134.,135.,136., \
#            137.,138.,139.,140.,141.,142.,143.,144.,145.,146.,147.,148.,149.,150.,151.,152.,153.,154.,155.,156.,157.,158., \
#            159.,160.,161.,162.,163.,164.,165.,166.,167.,168.,169.,169.1,169.2,169.3,169.4,169.5,169.6,169.7,169.8,169.9,170., \
#            170.1,170.2,170.3,170.4,170.5,170.6,170.7,170.8,170.9,171.,172.,173.,174.,175.,176.,177.,178.,179., \
#            179.2,179.4,179.6,179.8,179.9,179.99,179.999]

