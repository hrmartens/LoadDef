#!/usr/bin/env python

# *********************************************************************
# SPECIAL PROGRAM TO COMPUTE THE VERTICAL AND HORIZONTAL DISPLACEMENT 
#  RESPONSE TO A POLAR CAP LOADS USING AN ANALYTICAL APPROACH BASED ON
#  THE DISK FACTOR FROM FARRELL (1972); NO CONVOLUTION NECESSARY.
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
import matplotlib.pyplot as plt
import matplotlib.colorbar as clb
import matplotlib.cm as cm

# --------------- SPECIFY USER INPUTS --------------------- #
 
# Full path to Load Love Number file (output from run_ln.py)
#pmod = "PREM"
pmod = "Homogeneous_Vp05.92_Vs03.42_Rho03.00"
lln_file = ("../output/Love_Numbers/LLN/lln_" + pmod + ".txt")
 
# Apply a disk factor everywhere
diskf = True # True tells LoadDef to apply the disk factor
angdst = 0. # The angular distance from the load point at which to start applying the disk factor (for analytical disk, angdst must be set to 0.)

# Size of the disk
disks = 10. # The angular radius of the disk (in degrees)
print(':: Edge of disk in degrees: ', disks)

# Apply the disk load to both poles?
sym_caps = True

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

# Output filename (Default is 'grn.txt')
file_out = ("analyticalCaps_" + str(disks) + "deg-NoTaper_" + str(height_of_load) + "m_" + pmod + ".txt") 

# Synthesize symmetric cap results into output folder within current directory?
if sym_caps:
    write_syn = True

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


# Apply the disk loads to both poles? 
if sym_caps: 
    up_disp = u + np.flip(u)
    north_disp = v - np.flip(v) # for lateral displacements, we multiply the reverse by -1 to align with the convention of positive north
    # Multiply by 1000. to convert from meters to mm
    up_disp *= 1000.
    north_disp *= 1000.
    # Convert theta values to latitude
    theta_lat = np.asarray(theta)-90.
    # Write results to file
    if write_syn:
        if not (os.path.isdir("./output/")):
            os.makedirs("./output/")
        custom_file = ("./output/" + file_out)
        all_custom_data = np.column_stack((theta_lat,up_disp,north_disp))
        np.savetxt(custom_file, all_custom_data, fmt='%f %f %f')
    # Plot (as latitude)
    xmin = -90.
    xmax = 90.
    plt.subplot(2,1,1)
    plt.plot(theta_lat,north_disp,color='k',linestyle='-',linewidth=2)
    plt.xlim(xmin, xmax)
    plt.grid(True)
    plt.tick_params(labelsize='x-small')
    plt.title('North (mm)',size='small',weight='bold')
    plt.subplot(2,1,2)
    plt.plot(theta_lat,up_disp,color='k',linestyle='-',linewidth=2)
    plt.xlim(xmin, xmax)
    plt.grid(True)
    plt.xlabel(r'Latitude [$^{\circ}$] ',size='x-small')
    plt.tick_params(labelsize='x-small')
    plt.title('Up (mm)',size='small',weight='bold')
    plt.tight_layout()
    plt.show()

# --------------------- END CODE --------------------------- #

# Specify theta values (with symmetry; for symmetric polar caps)
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

# Specify theta values (asymmetric; better for high resolution of one disk)
# theta = [0.0001,0.0005,0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.082,0.084,0.086,0.088,0.089,0.0898,0.0899,0.09,0.091,0.092,0.094,0.096,0.098, \
#            0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.205,0.21,0.215,0.22,0.224,0.2245,0.2248,0.225,0.2255,0.226,0.23,0.235,0.24, \
#            0.25,0.26,0.27,0.28,0.29,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.44,0.46,0.48,0.5, \
#            0.6,0.8,1.0,2.,3.,4.,5.,6.,7.,8.,9.,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,10., \
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
 
