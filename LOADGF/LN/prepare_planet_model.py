# *********************************************************************
# FUNCTION TO PREPARE AN EARTH MODEL FOR INTEGRATION 
# INTERPOLATION, NON-DIMENSIONALIZATION
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

# Import Python Modules
from __future__ import print_function
import numpy as np
import math
import os
import sys
from scipy import interpolate
import matplotlib.pyplot as plt

# Import Modules from LoadDef
from LOADGF.utility import read_planet_model
from LOADGF.utility import convert_to_SI
from LOADGF.utility import compute_elastic_moduli
from LOADGF.utility import compute_gravity_nd
from LOADGF.utility import non_dimensionalize_parameters
from LOADGF.utility import interpolate_params
from LOADGF.utility import interpolate_planet_model

# Main Function
def main(emod_file,G=6.672E-11,r_min=1000.,kx=1,file_delim=None,emod_interp=False):

    # Read Earth Model
    r,vp,vs,rho = read_planet_model.main(emod_file,delim=file_delim)

    # If Model Starts at Surface, Reverse it
    if (r[0] > r[-1]):
        print(":: Reversing Earth model to start from core")
        r = r[::-1]
        vp = vp[::-1]
        vs = vs[::-1]
        rho = rho[::-1]

    # Convert to S.I. Units
    r,vp,vs,rho = convert_to_SI.main(r,vp,vs,rho)

    # Convert Seismic Velocities to Elastic Moduli
    mu,K,lmda = compute_elastic_moduli.main(vp,vs,rho)

    # Locate Discrete Regions (Discontinuities in Material Properties)
    region_index = np.where(np.diff(r) == 0.)
    region_index = region_index[0]
    # Add a Small Distance to the Upper Layer of the Interface
    if (len(region_index) > 0):
        small_dist = 0.1 # meters
        r[region_index+1] += small_dist

    # Interpolate Planet Model (Linearly) to a Fine Spacing
    #  :: Purpose: Starting Radii for the Integrations will be Determined Based on the Radial Array, So Coarse Models Could be Less Accurate and Less Consistent 
    #              Moreover, the Interpolation to Finer Layers Prevents Selecting a Starting Integration Radius at the Surface for Coarser Models Near the Surface
    if (emod_interp == True):
        print(':: Warning: Interpolating Earth model. Please verify results. [LOADGF/LN/prepare_planet_model.py]')
        # Coarse Interpolation
        coarse_interp = 50000.
        print((':: Interpolation: Coarse setting = ' + str(coarse_interp) + ' meters.'))
        rcs, mucs, Kcs, lmdacs, rhocs = interpolate_planet_model.main(r, mu, K, lmda, rho, coarse_interp, kx=kx, startinterp=min(r), stopinterp=max(r))
        r = np.concatenate((rcs,r))
        mu = np.concatenate((mucs,mu))
        K = np.concatenate((Kcs,K))
        lmda = np.concatenate((lmdacs,lmda))
        rho = np.concatenate((rhocs,rho))
        r,ridx = np.unique(r,return_index=True)
        mu = mu[ridx]
        K = K[ridx]
        lmda = lmda[ridx]
        rho = rho[ridx]
        # Fine Interpolation
        fine_interp = 10.
        fraction_fine_interp = 0.999
        fine_start = fraction_fine_interp*max(r)
        print((':: Interpolation: Fine setting = ' + str(fine_interp) + ' meters, applied to the upper ' + str(fraction_fine_interp) + ' fraction of input model.'))
        rfn,mufn,Kfn,lmdafn,rhofn = interpolate_planet_model.main(r, mu, K, lmda, rho, fine_interp, kx=kx, startinterp=fine_start, stopinterp=max(r))
        r = np.concatenate((rfn,r))
        mu = np.concatenate((mufn,mu))
        K = np.concatenate((Kfn,K))
        lmda = np.concatenate((lmdafn,lmda))
        rho = np.concatenate((rhofn,rho))
        r,ridx = np.unique(r,return_index=True)
        mu = mu[ridx]
        K = K[ridx]
        lmda = lmda[ridx]
        rho = rho[ridx]
    # Plot to Verify Interpolation
    #plt.plot(r,mu,color='k',linestyle='-')
    #plt.plot(r,mu,'bo',ms=2)
    #plt.show()
    #plt.plot(r,rho,color='k',linestyle='-')
    #plt.plot(r,rho,'bo',ms=2)
    #plt.show()

    # Non-Dimensionalize Parameters
    adim = max(r)                        # radius of Earth
    pi = np.pi
    piG = 1.                             # pi*G
    L_sc = adim                          # distance
    R_sc = 5500.                         # density
    T_sc = 1./(math.sqrt(R_sc*pi*G))     # time
    s,lnd,mnd,rnd = non_dimensionalize_parameters.main(r,lmda,rho,mu,L_sc,R_sc,T_sc)
    s_min = r_min/L_sc
    small = 1./L_sc # NOTE: May need to adjust if using a very small inner core size!

    # Compute Non-Dimensional Gravity
    g,gnd = compute_gravity_nd.main(s,rnd,piG,L_sc,T_sc)
    gsdim = g[-1]                        # gravity at surface

    # Compute Total Mass of Earth
    earth_radius = adim.copy()
    earth_mass = ((adim**2.)*gsdim)/G

    # Determine Radius of Inner and Outer Cores
    if any(mnd == 0):
        # Note: Current Procedure Can Only Handle One Fluid Layer (Not Multiple)
        ocind = np.where(mnd == 0); ocind = ocind[0]
        sic = s[min(ocind)]
        soc = s[max(ocind)]

        # Add new entries close to either side of the outer core (ensure that the interpolation does not set solid locations to near-zero)
        s = s.tolist(); lnd = lnd.tolist(); mnd = mnd.tolist(); rnd = rnd.tolist(); gnd = gnd.tolist()
        s.append(sic-small); s.append(soc+small)
        lnd.append(lnd[min(ocind)-1]); lnd.append(lnd[max(ocind)+1])
        mnd.append(mnd[min(ocind)-1]); mnd.append(mnd[max(ocind)+1]) 
        rnd.append(rnd[min(ocind)-1]); rnd.append(rnd[max(ocind)+1])
        gnd.append(gnd[min(ocind)-1]); gnd.append(gnd[max(ocind)+1])
        s = np.asarray(s); lnd = np.asarray(lnd); mnd = np.asarray(mnd); rnd = np.asarray(rnd); gnd = np.asarray(gnd)
        cidx = np.argsort(s)
        s = s[cidx]; lnd = lnd[cidx]; mnd = mnd[cidx]; rnd = rnd[cidx]; gnd = gnd[cidx]

    # Completely Solid 
    else:
        sic = 0.5462; soc = 0.5462 # No Fluid Outer Core; Set Inner Core Radius Equal to Outer Core Radius; Radius Should be Reasonable for Inner Core Change of Variables

    # Interpolate Non-Dimensional Parameters
    tck_lnd,tck_mnd,tck_rnd,tck_gnd = interpolate_params.main(s,lnd,mnd,rnd,gnd,kx=kx)

    # Return Variables
    return r,mu,K,lmda,rho,g,tck_lnd,tck_mnd,tck_rnd,tck_gnd,s,lnd,mnd,rnd,gnd,s_min,small,\
        earth_radius,earth_mass,sic,soc,adim,gsdim,pi,piG,L_sc,R_sc,T_sc


