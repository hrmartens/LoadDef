#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO GENERATE THE PREM MODEL FROM POLYNOMIAL FUNCTIONS
# LITERATURE: Dziewonski & Anderson (1981)
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

# MODIFY PYTHON PATH TO INCLUDE 'utility/emods' DIRECTORY
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd() + "/../")

# IMPORT PYTHON MODULES
import numpy as np
import matplotlib.pyplot as plt
from PREM import generate_PREM

# SPECIFY INTERVALS (km)
interval_cores = 100. 
interval_lowerMantle = 10.
interval_upperMantle = 5.
interval_crust = 1.

# SPECIFY OUTPUT FILE
outfile = ("./PREM_Isotropic.txt")

# GENERATE THE PREM MODEL
rdist,rho,vp,vs,Qmu,QK = generate_PREM.main(interval_cores,interval_lowerMantle,interval_upperMantle,interval_crust,outfile)

