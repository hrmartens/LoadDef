#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO PLOT LOAD GREEN'S FUNCTIONS (OUTPUT FROM run_ad.py)
#  HERE, PLOT THE OUTPUT FROM THE ANALYTICAL DISK COMPUTATION.
# 
# Copyright (c) 2014-2023: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
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
sys.path.append(os.getcwd() + "/../")

# Import Python Modules
import matplotlib.pyplot as plt
import matplotlib.colorbar as clb
import matplotlib.cm as cm
import numpy as np
 
# Filenames for Input Load Greens Function Files
#extension = "PREM_analyticalDisk"
extension = "Homogeneous_Vp05.92_Vs03.42_Rho03.00_analyticalDisk"
gfcm = ("../../output/Greens_Functions/cm_" + extension + ".txt")
gfce = ("../../output/Greens_Functions/ce_" + extension + ".txt")
gfcf = ("../../output/Greens_Functions/cf_" + extension + ".txt")

# Output Figure Name
figname1 = ("Greens_Functions_Displacement_" + extension + ".pdf")
figname2 = ("Greens_Functions_Displacement_SymCaps_" + extension + "_CE.pdf")
figname3 = ("Greens_Functions_Displacement_SymCaps_" + extension + "_CM.pdf") 
 
# X-axis Limits
xmin = 0
xmax = 180.

#### BEGIN CODE

# Create Folder
if not (os.path.isdir("./output/")):
    os.makedirs("./output/")
outdir = "./output/"
 
# Read the Files
thcm,ucm,vcm,ucmnorm,vcmnorm,gecm,tecm,ettcm,ellcm,gncm,tncm = np.loadtxt(gfcm,usecols=(0,1,2,3,4,5,6,8,9,10,11),unpack=True,skiprows=5)
thce,uce,vce,ucenorm,vcenorm,gece,tece,ettce,ellce,gnce,tnce = np.loadtxt(gfce,usecols=(0,1,2,3,4,5,6,8,9,10,11),unpack=True,skiprows=5)
thcf,ucf,vcf,ucfnorm,vcfnorm,gecf,tecf,ettcf,ellcf,gncf,tncf = np.loadtxt(gfcf,usecols=(0,1,2,3,4,5,6,8,9,10,11),unpack=True,skiprows=5)

# Convert from m to mm
ucm *= 1000.
vcm *= 1000.
uce *= 1000.
vce *= 1000.
ucf *= 1000.
vcf *= 1000.
 
# Plot Figure
fig1=plt.figure()
plt.subplot(2,3,1)
plt.plot(thcm[0:-1],vcm[0:-1],color='k',linestyle='-',linewidth=2)
plt.xlim(xmin, xmax)
#plt.xscale('log')
plt.grid(True)
#plt.ylabel(r'LGF [m/kg] $\times \, (10^{12} a \theta)$ [m]',size='x-small')
plt.ylabel(r'Displacement (mm)',size='x-small')
plt.tick_params(labelsize='x-small')
plt.title('Horizontal | CM',size='small',weight='bold')

plt.subplot(2,3,4)
plt.plot(thcm[0:-1],ucm[0:-1],color='k',linestyle='-',linewidth=2)
plt.xlim(xmin, xmax)
#plt.xscale('log')
plt.grid(True)
plt.xlabel(r'Distance to Load [$^{\circ}$] ',size='x-small')
plt.tick_params(labelsize='x-small')
plt.title('Vertical | CM',size='small',weight='bold')
#plt.ylabel(r'LGF [m/kg] $\times \, (10^{12} a \theta)$ [m]',size='x-small')
plt.ylabel(r'Displacement (mm)',size='x-small')

plt.subplot(2,3,2)
plt.plot(thce[0:-1],vce[0:-1],color='k',linestyle='-',linewidth=2)
plt.xlim(xmin, xmax)
#plt.xscale('log')
plt.grid(True)
plt.tick_params(labelsize='x-small')
plt.title('Horizontal | CE',size='small',weight='bold')

plt.subplot(2,3,5)
plt.plot(thce[0:-1],uce[0:-1],color='k',linestyle='-',linewidth=2)
plt.xlim(xmin, xmax)
plt.grid(True)
#plt.xscale('log')
plt.xlabel(r'Distance to Load [$^{\circ}$] ',size='x-small')
plt.tick_params(labelsize='x-small')
plt.title('Vertical | CE',size='small',weight='bold')

plt.subplot(2,3,3)
plt.plot(thcf[0:-1],vcf[0:-1],color='k',linestyle='-',linewidth=2)
plt.xlim(xmin, xmax)
#plt.xscale('log')
plt.grid(True)
plt.tick_params(labelsize='x-small')
plt.title('Horizontal | CF',size='small',weight='bold')

plt.subplot(2,3,6)
plt.plot(thcf[0:-1],ucf[0:-1],color='k',linestyle='-',linewidth=2)
plt.xlim(xmin, xmax)
plt.grid(True)
#plt.xscale('log')
plt.xlabel(r'Distance to Load [$^{\circ}$] ',size='x-small')
plt.tick_params(labelsize='x-small')
plt.title('Vertical | CF',size='small',weight='bold')

plt.tight_layout()
plt.savefig((outdir+figname1),orientation='landscape',format='pdf')
plt.show()

## Second figure for comparison with spherical caps
## Superpose the LGFs for one cap with its reverse (to simulate placing one cap at both poles)
uce_caps = uce + np.flip(uce)
vce_caps = vce - np.flip(vce)
plt.subplot(2,1,1)
plt.plot(thce,vce_caps,color='k',linestyle='-',linewidth=2)
plt.xlim(xmin, xmax)
plt.grid(True)
plt.tick_params(labelsize='x-small')
plt.title('Horizontal | CE',size='small',weight='bold')
plt.subplot(2,1,2)
plt.plot(thce,uce_caps,color='k',linestyle='-',linewidth=2)
plt.xlim(xmin, xmax)
plt.grid(True)
plt.xlabel(r'Distance to Load [$^{\circ}$] ',size='x-small')
plt.tick_params(labelsize='x-small')
plt.title('Vertical | CE',size='small',weight='bold')
plt.tight_layout()
plt.savefig((outdir+figname2),orientation='landscape',format='pdf')
plt.show()

## Second figure for comparison with spherical caps
## Superpose the LGFs for one cap with its reverse (to simulate placing one cap at both poles)
ucm_caps = ucm + np.flip(ucm)
vcm_caps = vcm - np.flip(vcm)
plt.subplot(2,1,1)
plt.plot(thcm,vcm_caps,color='k',linestyle='-',linewidth=2)
plt.xlim(xmin, xmax)
plt.grid(True)
plt.tick_params(labelsize='x-small')
plt.title('Horizontal | CM',size='small',weight='bold')
plt.subplot(2,1,2)
plt.plot(thcm,ucm_caps,color='k',linestyle='-',linewidth=2)
plt.xlim(xmin, xmax)
plt.grid(True)
plt.xlabel(r'Distance to Load [$^{\circ}$] ',size='x-small')
plt.tick_params(labelsize='x-small')
plt.title('Vertical | CM',size='small',weight='bold')
plt.tight_layout()
plt.savefig((outdir+figname3),orientation='landscape',format='pdf')
plt.show()

