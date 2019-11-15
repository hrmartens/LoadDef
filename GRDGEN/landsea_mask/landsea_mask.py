#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO GENERATE A SPARSE LAND-SEA MASK FROM ETOPO1
# https://www.ngdc.noaa.gov/mgg/global/global.html
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
from GRDGEN.utility import generate_landsea_mask

# --------------- SPECIFY USER INPUTS --------------------- #

# Directory and Name of ETOPO1 Grid File (Used to Generate the Land-Sea Mask)
landsea_dir = ("../../input/Land_Sea/")
etopo1_file = ("ETOPO1_Ice_g_gmt4.grd")

# Output Directory for Land-Sea Mask
outdir = ("../../output/Land_Sea/")
 
# Optional: Specifiy Path (and Filename) to the Antarctic Digital Database, in Ascii Format (Must Convert from Shapefile)
#           Generate using run_generate_antarctic_mask.py, in current directory.
antcst = (landsea_dir + "cst10srf_res0.05.txt")
#antcst = None # No ADD Coastlines
 
# Show figures?
show_figs = True

# ------------------ END USER INPUTS ----------------------- #

# -------------------- BEGIN CODE -------------------------- #

generate_landsea_mask.main(landsea_dir,etopo1_file,outdir,antarctic_coastline=antcst,show_figures=show_figs)

# --------------------- END CODE --------------------------- #


