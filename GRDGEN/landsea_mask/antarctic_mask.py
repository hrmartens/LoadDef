#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO GENERATE A SPARSE LAND-SEA MASK AROUND THE ANTARCTIC COASTLINE
# USING SHAPEFILES FROM THE ANTARCTIC DIGITAL DATABASE (ADD)
# http://www.add.scar.org/home/add7
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

# NOTE: Data files must be downloaded from the ADD website (requires registration)
# http://www.add.scar.org/home/add7

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd() + "/../../")

# Import Python Modules
from GRDGEN.utility import process_add_shapefile

# INPUT FILENAMES
landsea_dir = ("../../input/Land_Sea/")
#### Low Resolution
filename_shp  = (landsea_dir + "add_cst10_polygon/cst10_polygon.shp")
filename_dbf  = (landsea_dir + "add_cst10_polygon/cst10_polygon.dbf")
landsea_field = ("cst10srf")
#### Middle Resolution
#filename_shp  = (landsea_dir + "add_cst01_polygon/cst01_polygon.shp")
#filename_dbf  = (landsea_dir + "add_cst01_polygon/cst01_polygon.dbf")
#landsea_field = ("cst01srf")
#### High Resolution
#filename_shp  = (landsea_dir + "add_cst00_polygon/cst00_polygon.shp")
#filename_dbf  = (landsea_dir + "add_cst00_polygon/cst00_polygon.dbf")
#landsea_field = ("cst00srf") 

# Resolution of Grid (Degrees)
resolution = 0.05

# OUTPUT FILE (output directory = "../../output/Land_Sea/")
outfile = (landsea_field + "_res" + str(resolution) + ".txt")

# BEGIN CODE
process_add_shapefile.main(filename_shp, filename_dbf, landsea_field, outfile, resolution)

