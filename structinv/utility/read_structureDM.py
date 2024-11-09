# *********************************************************************
# FUNCTION TO READ IN A DESIGN MATRIX FOR SURFACE LOADING 
#   PURPOSE: READ IN THE FILE THAT WILL BE USED TO INVERT FOR STRUCTURE
# 
# Copyright (c) 2021-2024: HILARY R. MARTENS
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

import numpy as np
import scipy as sc
from math import pi
import sys
import netCDF4

def main(filename):

    # Read the netCDF file as a test
    f = netCDF4.Dataset(filename)
    #print(f.variables)
    sta_comp_ids = f.variables['sta_comp_id'][:]
    design_matrix = f.variables['design_matrix'][:]
    sta_comp_lat = f.variables['sta_comp_lat'][:]
    sta_comp_lon = f.variables['sta_comp_lon'][:]
    perturb_radius_bottom = f.variables['perturb_radius_bottom'][:]
    perturb_radius_top = f.variables['perturb_radius_top'][:]
    perturb_param = f.variables['perturb_param'][:]
    f.close()
 
    # Return Parameters
    return design_matrix,sta_comp_ids,sta_comp_lat,sta_comp_lon,perturb_radius_bottom,perturb_radius_top,perturb_param

