#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO GENERATE A HOMOGENEOUS EARTH MODEL
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

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd() + "/../../")

# IMPORT PYTHON MODULES
import numpy as np
from LOADGF.utility import read_planet_model
from LOADGF.utility import convert_to_SI
from LOADGF.utility import compute_elastic_moduli

# Output Filename
outfile = ("Homogeneous.txt")

# Material Properties of the Homogeneous Sphere
p_velocity = 10. # km/s
s_velocity = 5.  # km/s
density    = 5.  # g/cc

# Radius of Homogeneous Sphere
radius = 6371.   # km

# Number of Radial Steps to Write Out
num_steps = 100.

# Optionally Add a Small Fluid Layer?
small_fluid = False

# BEGIN CODE

# Ensure that the Output Directories Exist
if not (os.path.isdir("../../output/Planet_Models/")):
    os.makedirs("../../output/Planet_Models")

# Radial Steps
radial_dist = np.linspace(0.,radius,num=num_steps,endpoint=True)

# Generate Homogeneous Model
vp = np.ones(len(radial_dist),)*p_velocity
vs = np.ones(len(radial_dist),)*s_velocity
rho = np.ones(len(radial_dist),)*density

# Optionally Add a Small Fluid Layer (for testing)
if (small_fluid == True):
    # Add a Few Radial Steps to the Radial Distance Array
    radial_dist = radial_dist.tolist() + [149.9, 150., 151., 151.1]
    radial_dist = np.sort(np.asarray(radial_dist))
    # Generate Homogeneous Model
    vp = np.ones(len(radial_dist),)*p_velocity
    vs = np.ones(len(radial_dist),)*s_velocity
    rho = np.ones(len(radial_dist),)*density
    # Set vs=0 for the (Very Small) Fluid Layer
    flidx = np.where(radial_dist == 150.); flidx = flidx[0]
    vs[flidx] = 0.; vs[flidx+1] = 0.

# Write to File
fname = ("../../output/Planet_Models/" + outfile)
params = np.column_stack((radial_dist,vp,vs,rho))
#f_handle = open(fname,'w')
np.savetxt(fname,params,fmt='%f %f %f %f')
#f_handle.close() 

