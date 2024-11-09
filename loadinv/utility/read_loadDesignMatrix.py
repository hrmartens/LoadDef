# *********************************************************************
# FUNCTION TO READ IN A JACOBIAN MATRIX FOR SURFACE LOADING
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

    # Read in Data File (netCDF format)
    f = netCDF4.Dataset(filename)
    #print(f.variables)
    sta_comp_ids = f.variables['sta_comp_id'][:]
    load_cell_ids = f.variables['load_cell_id'][:]
    design_matrix = f.variables['design_matrix'][:]
    sta_comp_lat = f.variables['sta_comp_lat'][:]
    sta_comp_lon = f.variables['sta_comp_lon'][:]
    load_cell_lat = f.variables['load_cell_lat'][:]
    load_cell_lon = f.variables['load_cell_lon'][:]
    f.close()
    #print(sta_comp_ids)
    #print(load_cell_ids)
    #print(design_matrix)
    #print(sta_comp_lat)
    #print(sta_comp_lon)
    #print(load_cell_lat)
    #print(load_cell_lon)
 
    # Return Parameters
    return design_matrix,sta_comp_ids,sta_comp_lat,sta_comp_lon,load_cell_ids,load_cell_lat,load_cell_lon

