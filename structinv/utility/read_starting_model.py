# *********************************************************************
# FUNCTION TO READ IN STARTING MODEL of SURFACE DISPLACEMENT
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
import sys

def main(filename,inc_imag=False):

    # Read in station names
    sta = np.loadtxt(filename,usecols=(0,),unpack=True,dtype='U',skiprows=1)
   
    # Read in model
    if (inc_imag == True): 
        lat,lon,ere,nre,ure,eim,nim,uim = np.loadtxt(filename,usecols=(1,2,3,4,5,6,7,8),unpack=True,skiprows=1)
    else:
        lat,lon,edisp,ndisp,udisp = np.loadtxt(filename,usecols=(1,2,3,4,5),unpack=True,skiprows=1)
 
    # Return Parameters
    if (inc_imag == True):
        return sta,lat,lon,ere,eim,nre,nim,ure,uim
    else:
        return sta,lat,lon,edisp,ndisp,udisp

