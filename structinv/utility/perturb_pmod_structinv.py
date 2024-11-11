#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO PERTURB THE PARAMETERS OF AN EARTH MODEL
# POTENTIAL APPLICATION: DEVELOP SENSITIVITY KERNELS FOR LOAD GREEN'S FUNCTIONS
# LITERATURE: Martens et al. (2016, JGR-Solid Earth)
#
# ALLOW PERTURBATIONS TO VARY WITH DEPTH
# 
# Copyright (c) 2023-2024: HILARY R. MARTENS
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
def main(planet_model,pert_param,pert_rad_bot,pert_rad_top,cmv,ref_rho=1,ref_mu=1,ref_ka=1,include_start_point=False,include_end_point=True):

    # Prepare Planet Model
    print(':: Preparing Planet Model')
    radial_dist,mu_orig,ka_orig,lmda_orig,rho_orig,g,tck_lnd,tck_mnd,tck_rnd,tck_gnd,s,lnd,mnd,rnd,gnd,s_min,small,\
        planet_radius,planet_mass,sic,soc,adim,gsdim,pi,piG,L_sc,R_sc,T_sc = prepare_planet_model.main(planet_model)

    # Make copies of the original arrays
    mu_new = mu_orig.copy()
    ka_new = ka_orig.copy()
    rho_new = rho_orig.copy()

    # Convert All Material Parameters to Log Space 
    log_rho_orig = np.log10(rho_orig/ref_rho)
    log_ka_orig = np.log10(ka_orig/ref_ka)
    # Set Shear Modulus in Outer Core (Fluid) Region to a Very Small Number Just Slightly Above Zero (A Perfectly Zero Shear Modulus is Unphysical, and Won't Work in Log Space)
    zero_idx = np.where(mu_orig == 0); zero_idx = zero_idx[0]
    mu_orig[zero_idx] = 1E-17
    log_mu_orig = np.log10(mu_orig/ref_mu)

    # Loop through parameters and radii ranges to perturb
    for ee in range(0,len(pert_param)):

        # Current parameter
        cparam = pert_param[ee]
        # Current bottom radius
        cbotrad = pert_rad_bot[ee]
        # Current top radius 
        ctoprad = pert_rad_top[ee]
        # Current perturbation
        cpert = cmv[ee]
        # Print parameters
        #print(cparam)
        #print(cbotrad)
        #print(ctoprad)
        #print(cpert)

        # Find Indices to Perturb
        print(':: Isolating Indices to Perturb [perturb_pmod_structinv.py]')
        myidx = np.where((radial_dist >= cbotrad*1000.) & (radial_dist <= ctoprad*1000.)); myidx = myidx[0]
        if (len(myidx) == 0):
            print(':: Warning: No Points Found in Depth Range.')
            continue
        if (include_start_point == False):
            myidx = myidx[1::]
        if (include_end_point == False):
            myidx = myidx[0:-1]
        print(myidx)

        # Perform the perturbations
        if (cparam == 'mu'): 
            # Perturb Mu and Convert Back to Linear Space
            log_mu_new = (log_mu_orig[myidx] + cpert)
            mu_new[myidx] = np.power(10.,log_mu_new)*ref_mu
            #print(mu_orig[myidx])
            #print(mu_new[myidx])
            #print(log_mu_orig[myidx])
            #print(log_mu_new)
            #sys.exit()
        elif (cparam == 'kappa'):
            # Perturb Kappa and Convert Back to Linear Space
            log_ka_new = (log_ka_orig[myidx] + cpert)
            ka_new[myidx] = np.power(10.,log_ka_new)*ref_ka
            #print(ka_orig[myidx])
            #print(ka_new[myidx])
            #print(log_ka_orig[myidx])
            #print(log_ka_new)
            #sys.exit()
        elif (cparam == 'rho'):
            # Perturb Rho and Convert Back to Linear Space
            log_rho_new = (log_rho_orig[myidx] + cpert)
            rho_new[myidx] = np.power(10.,log_rho_new)*ref_rho
            #print(rho_orig[myidx])
            #print(rho_new[myidx])
            #print(log_rho_orig[myidx])
            #print(log_rho_new)
            #sys.exit()
        else: 
            sys.exit(':: Error: Undefined parameter.')

        # Convert fluid regions back to mu=0
        zero_idx = np.where(mu_new < 1E-15); zero_idx = zero_idx[0]
        mu_new[zero_idx] = 0.

    # Return variables
    return radial_dist, mu_orig, ka_orig, rho_orig, mu_new, ka_new, rho_new

