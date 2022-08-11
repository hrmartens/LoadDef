#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO CONVERT HARMONIC DISPLACEMENTS (OUTPUT FROM run_cn.py)
#   TO PARTICLE MOTION ELLIPSES (PMEs)
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
from CONVGF.utility import env2pme
from CONVGF.utility import read_convolution_file
import numpy as np

#### USER INPUT ####

harmonic="M2"
output_directory = ("./output/")
filename=(output_directory + "cn_OceanOnly_" + harmonic + "_cm_convgf_GOT410c_PREM.txt")
pme_file=(output_directory + "pme_OceanOnly_" + harmonic + "_cm_convgf_GOT410c_PREM.txt")

#### BEGIN CODE ####

# Create output directory, if it does not yet exist
if not (os.path.isdir(output_directory)):
    os.makedirs(output_directory)

sta,lat,lon,eamp,epha,namp,npha,vamp,vpha = read_convolution_file.main(filename)

# Perform the Conversion
smmjr,smmnr,theta = env2pme.main(eamp,epha,namp,npha)

# Force Theta Positive
theta[theta < 0.] += 360.

# Remove Duplicate Stations
unique_sta, usta_idx = np.unique(sta,return_index=True)
sta = sta[usta_idx]; lat = lat[usta_idx]; lon = lon[usta_idx]; eamp = eamp[usta_idx]; epha = epha[usta_idx]
namp = namp[usta_idx]; npha = npha[usta_idx]; vamp = vamp[usta_idx]; vpha = vpha[usta_idx]; smmjr = smmjr[usta_idx]
smmnr = smmnr[usta_idx]; theta = theta[usta_idx]

# Prepare Output Files
pme_head = ("../../output/Convolution/pme_head.txt")
pme_body = ("../../output/Convolution/pme_body.txt")

# Prepare Data for Output (as Structured Array)
all_pme_data = np.array(list(zip(sta,lat,lon,theta,smmjr,smmnr,eamp,epha,namp,npha,vamp,vpha)), dtype=[('sta','U25'), \
    ('lat',float),('lon',float),('theta',float),('smmjr',float),('smmnr',float),('eamp',float),('epha',float), \
    ('namp',float),('npha',float),('vamp',float),('vpha',float)])

# Write Header Info to File
hf = open(pme_head,'w')
pme_str = 'Station  Lat(+N,deg)  Lon(+E,deg)  Direction(deg)  Semi-Major(mm)  Semi-Minor(mm)  E-Amp(mm)  E-Pha(deg)  N-Amp(mm)  N-Pha(deg)  V-Amp(mm)  V-Pha(deg) \n'
hf.write(pme_str)
hf.close()

# Write PME Results to File
#f_handle = open(pme_body,'w')
np.savetxt(pme_body,all_pme_data,fmt=["%s"]+["%.8f",]*11,delimiter="        ")
#f_handle.close()

# Combine Header and Body Files
filenames_pme = [pme_head, pme_body]
with open(pme_file,'w') as outfile:
    for fname in filenames_pme:
        with open(fname) as infile:
            outfile.write(infile.read())
 
# Remove Header and Body Files
os.remove(pme_head)
os.remove(pme_body)


