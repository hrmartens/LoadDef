# *********************************************************************
# FUNCTION TO READ IN A LOADDEF SURFACE-DISPLACEMENT (CONVOLUTION) FILE
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

import numpy as np

def main(filename):
    lat,lon,eamp,epha,namp,npha,vamp,vpha = np.loadtxt(filename,usecols=(1,2,3,4,5,6,7,8),unpack=True,skiprows=1)
    sta = np.loadtxt(filename,usecols=(0,),dtype='U',unpack=True,skiprows=1)
    return sta,lat,lon,eamp,epha,namp,npha,vamp,vpha

