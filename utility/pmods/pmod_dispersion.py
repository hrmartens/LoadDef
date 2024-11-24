#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO ADJUST THE ELASTIC PARAMETERS OF AN EARTH MODEL
# REFERENCED TO 1-SECOND PERIOD TO A DIFFERENT REFERENCE PERIOD
# (e.g. M2 TIDAL PERIOD) | ACCOUNTS FOR PHYSICAL DISPERSION
# 
# Copyright (c) 2014-2024: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
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

# IMPORT PYTHON MODULES
from __future__ import print_function
import numpy as np
from math import pi
import os
import sys

# Reference Period to Which the Model Should be Adjusted
rper = 12.42*(60.*60.)

# Input Planet Model
#  :: Input Planet Model Must be of the Format: 
#  :: Radius[km], vp[km/s], vs[km/s], density[g/cc], Qk, Qmu
planet_model = ("../../input/Planet_Models/PREM.txt")

# Output Filename
outfile = ("PREM_M2TidalPeriod.txt")

# BEGIN CODE

# Ensure that the Output Directories Exist
if not (os.path.isdir("../../output/Planet_Models/")):
    os.makedirs("../../output/Planet_Models/")
 
# Read Reference Planet Model
radial_dist,vp,vs,rho,Qk,Qmu = np.loadtxt(planet_model,usecols=(0,1,2,3,4,5),unpack=True)
mu = np.multiply(np.square(vs*1000),rho*1000)
ka = np.multiply(np.square(vp*1000),rho*1000) - (4./3.)*mu
# Convert to Vp,Vs
vs = np.sqrt(np.divide(mu,rho*1000.)) / 1000.
vp = np.sqrt(np.divide((ka + (4./3.)*mu),rho*1000.)) / 1000.
print(('mu[orig]', mu))
print(('ka[orig]', ka))
print(('vp[orig]', vp))
print(('vs[orig]', vs))

# Convert to Vp,Vs at Tidal Frequency (e.g. Stein and Wysession, Eqs. 51 & 52)
E = (4./3.)*np.square(np.divide(vs,vp))
coef = np.log(rper)/pi
QmuInv = 1./Qmu
QkInv = 1./Qk
test = np.isinf(QmuInv); mynan = np.where(test == 1); mynan = mynan[0]
QmuInv[test] = 0.
argvs = (1.-coef*QmuInv)
argvp = (1. - coef*(np.multiply((1.-E),QkInv) + np.multiply(E,QmuInv)))
vs_pert = np.multiply(vs , argvs)
vp_pert = np.multiply(vp , argvp)
mu_pert = np.multiply(np.square(vs_pert*1000),rho*1000)
ka_pert = np.multiply(np.square(vp_pert*1000),rho*1000) - (4./3.)*mu_pert
print(('mu[S&W]', mu_pert))
print(('ka[S&W]', ka_pert))
print(('vp[S&W]', vp_pert))
print(('vs[S&W]', vs_pert))

# Verify Against Mu at Tidal Frequencies (e.g. Bos et al 2015, Eq. 3; Dahlen & Tromp, Eqs. 9.48 and 9.49)
dmu = np.multiply(mu,QmuInv)*((2./pi)*np.log(1/rper))
newmu = mu+dmu # Note that 'newmu' should be ~ equivalent to 'mu_pert'
dka = np.multiply(ka,QkInv)*((2./pi)*np.log(1./rper))
newka = ka+dka
print(('mu[D&T]', newmu))
print(('ka[D&T]', newka))
vp_pert = np.sqrt(np.divide((newka + (4./3.)*newmu),rho*1000)) / 1000.
vs_pert = np.sqrt(np.divide((newmu),rho*1000)) / 1000.
print(('vp[D&T]', vp_pert))
print(('vs[D&T]', vs_pert))

# Write to File
fname = ("../../output/Planet_Models/" + outfile)
print(':: Output file path: ', fname)
params = np.column_stack((radial_dist,vp_pert,vs_pert,rho,Qk,Qmu))
#f_handle = open(fname,'w')
np.savetxt(fname,params,fmt='%f %f %f %f %f %f')
#f_handle.close()

