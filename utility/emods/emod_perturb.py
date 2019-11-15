#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO PERTURB THE PARAMETERS OF AN EARTH MODEL
# POTENTIAL APPLICATION: DEVELOP SENSITIVITY KERNELS FOR LOAD GREEN'S FUNCTIONS
# LITERATURE: Martens et al. (2016, JGR-Solid Earth)
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

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd() + "/../../")

# IMPORT PYTHON MODULES
import numpy as np
from LOADGF.LN import prepare_earth_model

# Specify the Perturbation Amount  
#  :: The Perturbation Factor Will be Added to the Log of Each Model Parameter
m_perturb = np.log10(1.01)

# Specify the Perturbation Depth Range
d_perturb = [6291.,6346.6] # km

# Would you Like to Include the Start and End Points?
include_start_point = False # First Point for Lower Bound
include_end_point = True # Last Point for Upper Bound

# Input and Output Filenames
earth_model = ("../../input/Planet_Models/PREM.txt")
outfile = ("PREM_" + str('{:.4f}'.format(m_perturb)) + "_" + str(d_perturb[0]) + "_" + str(d_perturb[1]) + ".txt")

#### BEGIN CODE

# Ensure that the Output Directories Exist
if not (os.path.isdir("../../output/Planet_Models/")):
    os.makedirs("../../output/Planet_Models/")
outdir = "../../output/Planet_Models/" 

# Prepare Planet Model
print(':: Preparing Planet Model')
radial_dist,mu,kappa,lmda,rho,g,tck_lnd,tck_mnd,tck_rnd,tck_gnd,s,lnd,mnd,rnd,gnd,s_min,small,\
    earth_radius,earth_mass,sic,soc,adim,gsdim,pi,piG,L_sc,R_sc,T_sc = prepare_earth_model.main(earth_model)
  
# Convert to Vp,Vs
vs = np.sqrt(np.divide(mu,rho))
vp = np.sqrt(np.divide((kappa + (4./3.)*mu),rho))

# Convert All Material Parameters to Log Space 
log_rho = np.log10(rho)
log_kappa = np.log10(kappa)

# Set Shear Modulus in Outer Core (Fluid) Region to a Very Small Number Just Slightly Above Zero (A Perfectly Zero Shear Modulus is Unphysical, and Won't Work in Log Space)
zero_idx = np.where(mu == 0); zero_idx = zero_idx[0]
mu[zero_idx] = 1E-17
log_mu = np.log10(mu)

# Find Indices to Perturb
print(':: Isolating Indices to Perturb')
myidx = np.where((radial_dist >= d_perturb[0]*1000.) & (radial_dist <= d_perturb[1]*1000.)); myidx = myidx[0]
if (len(myidx) == 0):
    sys.exit('Error: No Points Found in Depth Range.')
if (include_start_point == False):
    myidx = myidx[1::]
if (include_end_point == False):
    myidx = myidx[0:-1]

#### 1. Perturb Mu
print(':: Perturbing Mu')
fname = (outdir + "mu_" + outfile)
log_mu_pert = log_mu.copy()
log_mu_pert[myidx] += m_perturb
# Convert Back to Linear Space
mu_pert = np.power(10.,log_mu_pert)
zero_idx = np.where(mu_pert < 1E-15); zero_idx = zero_idx[0]
mu_pert[zero_idx] = 0.
kappa = np.power(10.,log_kappa)
rho = np.power(10.,log_rho)
# Convert to Vp,Vs,Rho
vs_pert = np.sqrt(np.divide(mu_pert,rho)) 
vp_pert = np.sqrt(np.divide((kappa + (4./3.)*mu_pert),rho)) 
# Write to File
params = np.column_stack((radial_dist/1000.,vp_pert/1000.,vs_pert/1000.,rho/1000.))
#f_handle = open(fname,'w')
np.savetxt(fname,params,fmt='%f %f %f %f')
#f_handle.close() 
 
#### 2. Perturb Kappa
print(':: Perturbing Kappa')
fname = (outdir + "ka_" + outfile)
log_kappa_pert = log_kappa.copy()
log_kappa_pert[myidx] += m_perturb
# Convert Back to Linear Space
mu = np.power(10.,log_mu)
zero_idx = np.where(mu < 1E-15); zero_idx = zero_idx[0]
mu[zero_idx] = 0.
kappa_pert = np.power(10.,log_kappa_pert)
rho = np.power(10.,log_rho)
# Convert to Vp,Vs,Rho
vs_pert = np.sqrt(np.divide(mu,rho))
vp_pert = np.sqrt(np.divide((kappa_pert + (4./3.)*mu),rho))
# Write to File
params = np.column_stack((radial_dist/1000.,vp_pert/1000.,vs_pert/1000.,rho/1000.))
#f_handle = open(fname,'w')
np.savetxt(fname,params,fmt='%f %f %f %f')
#f_handle.close()

#### 3. Perturb Rho (Mu and Kappa Held Constant)
print(':: Perturbing Density')
fname = (outdir + "rho_" + outfile)
log_rho_pert = log_rho.copy()
log_rho_pert[myidx] += m_perturb
# Convert Back to Linear Space
mu = np.power(10.,log_mu)
zero_idx = np.where(mu < 1E-15); zero_idx = zero_idx[0]
mu[zero_idx] = 0.
kappa = np.power(10.,log_kappa)
rho_pert = np.power(10.,log_rho_pert)
# Convert to Vp,Vs,Rho
vs_pert = np.sqrt(np.divide(mu,rho_pert))
vp_pert = np.sqrt(np.divide((kappa + (4./3.)*mu),rho_pert))
# Write to File
params = np.column_stack((radial_dist/1000.,vp_pert/1000.,vs_pert/1000.,rho_pert/1000.))
#f_handle = open(fname,'w')
np.savetxt(fname,params,fmt='%f %f %f %f')
#f_handle.close()

#### 4. Perturb Rho (Vp and Vs held constant)
print(':: Perturbing Density (with P- and S-wave Velocities Held Constant)')
fname = (outdir + "rho_VpVsConstant_" + outfile)
log_rho_pert = log_rho.copy()
log_rho_pert[myidx] += m_perturb
# Convert Back to Linear Space
mu = np.power(10.,log_mu)
zero_idx = np.where(mu < 1E-15); zero_idx = zero_idx[0]
mu[zero_idx] = 0.
kappa = np.power(10.,log_kappa)
rho_pert = np.power(10.,log_rho_pert)
# Write to File
params = np.column_stack((radial_dist/1000.,vp/1000.,vs/1000.,rho_pert/1000.))
#f_handle = open(fname,'w')
np.savetxt(fname,params,fmt='%f %f %f %f')
#f_handle.close()

