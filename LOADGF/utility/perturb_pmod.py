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

# IMPORT PYTHON MODULES
import numpy as np
import os
import sys
from LOADGF.LN import prepare_planet_model

#  :: planet_model: File containing the model for planetary structure [radius (km), Vp (km/s), Vs (km/s), Density (g/cc)]
#  :: pmod: Name of planetary model (for output filenames)
#  :: perturb: The Perturbation Factor Will be Added to the Log of Each Model Parameter 
#  :: rad_range: The range of radii to perturb (in km)
#  :: include_start_point: Include first point for lower bound?
#  :: include_end_point: Include last point for upper bound?
def main(planet_model,pmod,perturb,rad_range,outdir,ref_rho=1,ref_mu=1,ref_ka=1,include_start_point=False,include_end_point=True,suffix=""):

    # Output Filename
    outname = (str('{:.4f}'.format(perturb)) + "_" + str(rad_range[0]) + "_" + str(rad_range[1]) + suffix)

    # Prepare Planet Model
    print(':: Preparing Planet Model')
    radial_dist,mu,kappa,lmda,rho,g,tck_lnd,tck_mnd,tck_rnd,tck_gnd,s,lnd,mnd,rnd,gnd,s_min,small,\
        planet_radius,planet_mass,sic,soc,adim,gsdim,pi,piG,L_sc,R_sc,T_sc = prepare_planet_model.main(planet_model)
   
    # Convert All Material Parameters to Log Space 
    log_rho = np.log10(rho/ref_rho)
    log_kappa = np.log10(kappa/ref_ka)
    # Set Shear Modulus in Outer Core (Fluid) Region to a Very Small Number Just Slightly Above Zero (A Perfectly Zero Shear Modulus is Unphysical, and Won't Work in Log Space)
    zero_idx = np.where(mu == 0); zero_idx = zero_idx[0]
    mu[zero_idx] = 1E-17
    log_mu = np.log10(mu/ref_mu)

    # Find Indices to Perturb
    print(':: Isolating Indices to Perturb')
    myidx = np.where((radial_dist >= rad_range[0]*1000.) & (radial_dist <= rad_range[1]*1000.)); myidx = myidx[0]
    if (len(myidx) == 0):
        sys.exit('Error: No Points Found in Depth Range.')
    if (include_start_point == False):
        myidx = myidx[1::]
    if (include_end_point == False):
        myidx = myidx[0:-1]

    #### 1. Perturb Mu
    print(':: Perturbing Mu')
    mu_name = (pmod + "_mu_" + outname)
    fname_mu = (outdir + mu_name + ".txt")
    log_mu_pert = log_mu.copy()
    log_mu_pert[myidx] += perturb
    # Convert Back to Linear Space
    mu_pert = np.power(10.,log_mu_pert)*ref_mu
    zero_idx = np.where(mu_pert < 1E-15); zero_idx = zero_idx[0]
    mu_pert[zero_idx] = 0.
    kappa = np.power(10.,log_kappa)*ref_ka
    rho = np.power(10.,log_rho)*ref_rho
    # Convert to Vp,Vs,Rho
    vs_pert = np.sqrt(np.divide(mu_pert,rho)) 
    vp_pert = np.sqrt(np.divide((kappa + (4./3.)*mu_pert),rho)) 
    # Write to File
    params = np.column_stack((radial_dist/1000.,vp_pert/1000.,vs_pert/1000.,rho/1000.))
    np.savetxt(fname_mu,params,fmt='%f %f %f %f')
 
    #### 2. Perturb Kappa
    print(':: Perturbing Kappa')
    ka_name = (pmod + "_kappa_" + outname)
    fname_ka = (outdir + ka_name + ".txt")
    log_kappa_pert = log_kappa.copy()
    log_kappa_pert[myidx] += perturb
    # Convert Back to Linear Space
    mu = np.power(10.,log_mu)*ref_mu
    zero_idx = np.where(mu < 1E-15); zero_idx = zero_idx[0]
    mu[zero_idx] = 0.
    kappa_pert = np.power(10.,log_kappa_pert)*ref_ka
    rho = np.power(10.,log_rho)*ref_rho
    # Convert to Vp,Vs,Rho
    vs_pert = np.sqrt(np.divide(mu,rho))
    vp_pert = np.sqrt(np.divide((kappa_pert + (4./3.)*mu),rho))
    # Write to File
    params = np.column_stack((radial_dist/1000.,vp_pert/1000.,vs_pert/1000.,rho/1000.))
    np.savetxt(fname_ka,params,fmt='%f %f %f %f')

    #### 3. Perturb Rho (Mu and Kappa Held Constant)
    print(':: Perturbing Density')
    rho_name = (pmod + "_rho_" + outname)
    fname_rho = (outdir + rho_name + ".txt")
    log_rho_pert = log_rho.copy()
    log_rho_pert[myidx] += perturb
    # Convert Back to Linear Space
    mu = np.power(10.,log_mu)*ref_mu
    zero_idx = np.where(mu < 1E-15); zero_idx = zero_idx[0]
    mu[zero_idx] = 0.
    kappa = np.power(10.,log_kappa)*ref_ka
    rho_pert = np.power(10.,log_rho_pert)*ref_rho
    # Convert to Vp,Vs,Rho
    vs_pert = np.sqrt(np.divide(mu,rho_pert))
    vp_pert = np.sqrt(np.divide((kappa + (4./3.)*mu),rho_pert))
    # Write to File
    params = np.column_stack((radial_dist/1000.,vp_pert/1000.,vs_pert/1000.,rho_pert/1000.))
    np.savetxt(fname_rho,params,fmt='%f %f %f %f')

    # Return variables
    return fname_mu, fname_ka, fname_rho, mu_name, ka_name, rho_name

