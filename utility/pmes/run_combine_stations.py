#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO COMBINE STATION FILES INTO INDIVIDUAL HARMONIC FILES 
#   FOR THE NETWORK
# PURPOSE: CONVERT EAST and NORTH AMPLITUDES TO HORIZONTAL PMEs
# LITERATURE: Martens et al. (2016, GJI), Martens (2016, Caltech)
# 
# Copyright (c) 2014-2022: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
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

# Import Python Modules
from CONVGF.utility import read_convolution_file
from utility.pmes import combine_stations
import numpy as np

#### USER INPUT ####

directory = ("../../output/Convolution/")
prefix = ("cn_LandAndOceans_")
suffix = ("ce_convgf_disk_1m_PREM.txt") 

#### BEGIN CODE ####

combine_stations.main(directory,prefix,suffix)

