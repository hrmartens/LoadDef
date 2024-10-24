# *********************************************************************
# FUNCTION TO INTEGRATE ODEs for SPHEROIDAL DEFORMATION OF AN ELASTIC BODY
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

import numpy as np
import math
import sys
from LOADGF.LN import integrate_fullEarth_n0
from LOADGF.LN import integrate_fullEarth
from LOADGF.LN import integrate_mantle
from LOADGF.LN import apply_boundary_conditions_n0
from LOADGF.LN import apply_boundary_conditions
from LOADGF.LN import evaluate_load_ln_n0
from LOADGF.LN import evaluate_load_ln
from LOADGF.LN import evaluate_potential_ln_n0
from LOADGF.LN import evaluate_potential_ln
from LOADGF.LN import evaluate_stress_ln_n0
from LOADGF.LN import evaluate_stress_ln
from LOADGF.LN import evaluate_shear_ln_n0
from LOADGF.LN import evaluate_shear_ln
from LOADGF.LN import compute_solutions_n0
from LOADGF.LN import compute_solutions
from LOADGF.LN import fundamental_solutions_homsph
from LOADGF.LN import fundamental_solutions_homsph_n0
from scipy import interpolate
 
def main(n,s_min,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,sic,soc,small,\
    num_soln,backend,abs_tol,rel_tol,nstps,order,gnd,adim,gsdim,L_sc,T_sc,inf_tol,s,nmaxfull,kx=1,eval_radii=1,numrad=1):

    # Determine the maximum n for which whole-planet integration will be performed (if not specified by user)
    if nmaxfull is None:
        # Determine When to Integrate Through Cores, Versus When to Begin Integration in the Mantle (Note: r^n > inf_tol)
        # (r/a)^n > inf_tol : start integration at the r for which this expression holds
        # Logarithm power rule: log((r/a)^n) = n log(r/a) = log(inf_tol)
        # soc = roc/a, where roc = radius of the outer core
        # Thus: n = log(inf_tol)/log(soc) yields the n above which integrations should start in the mantle
        # To be safe, we round upward, and any n beneath that value will be integrated through the whole Earth
        # Note: If receiving an error like this: ValueError: operands could not be broadcast together with shapes (2,) (100,) 
        #       then consider further limiting the number of spherical-harmonic degrees integrated through full body
        nmaxfull = math.ceil(math.log(inf_tol)/math.log(soc))

    # Initialize arrays (when computing Love numbers at more than one depth)
    hprime = np.empty(numrad)
    nlprime = np.empty(numrad)
    nkprime = np.empty(numrad)
    hpot = np.empty(numrad)
    nlpot = np.empty(numrad)
    nkpot = np.empty(numrad)
    hstr = np.empty(numrad)
    nlstr = np.empty(numrad)
    nkstr = np.empty(numrad)
    hshr = np.empty(numrad)
    nlshr = np.empty(numrad)
    nkshr = np.empty(numrad)
    sidx = np.empty(numrad)
  
    # Special Case: n=0
    if (n == 0):

        # Perform the Integration Through Full Earth
        Y1, Y2, sint_mt = integrate_fullEarth_n0.main(n,s_min,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,sic,soc,small,\
            num_soln,backend,abs_tol,rel_tol,nstps,order,adim,gsdim,L_sc,T_sc,inf_tol,s)
 
        # Compute Solutions For Homogeneous Sphere (Size of Earth, for Different n)
        Y1_shape = np.asarray(Y1).shape
        Y = fundamental_solutions_homsph_n0.main(1.,n,tck_lnd,tck_mnd,tck_rnd,wnd,piG)
        Y1i = Y[0,:]
        Y2i = Y[1,:]
        Y1 = np.ones((Y1_shape[0],Y1_shape[1]))*Y1i
        Y2 = np.ones((Y1_shape[0],Y1_shape[1]))*Y2i
        print(':: Reminder: Computing Y Solutions Based on Homogeneous Sphere.')

        # Apply Boundary Conditions at Surface
        m_load,m_pot,m_str,m_shr = apply_boundary_conditions_n0.main(n,Y1[-1],Y2[-1],gnd[-1],piG)

        # Compute Y Solutions
        Y_load = compute_solutions_n0.main(Y1,Y2,m_load)
        Y_pot  = compute_solutions_n0.main(Y1,Y2,m_pot)
        Y_str  = compute_solutions_n0.main(Y1,Y2,m_str)
        Y_shr  = compute_solutions_n0.main(Y1,Y2,m_shr)

        # Compute Love Numbers at Desired Depth: Find the appropriate radii
        if (numrad == 1):
            abs_val_diff = np.abs(np.asarray(sint_mt) - float(eval_radii)) # absolute values of the differences between the integration radii and the test depth
            sidx = abs_val_diff.argmin() # the index for the desired depth is found by locating the smallest value in the difference array
        else:
            for jj in range(0,numrad):
                abs_val_diff = np.abs(np.asarray(sint_mt) - float(eval_radii[jj])) # absolute values of the differences between the integration radii and the test depth
                sidx[jj] = abs_val_diff.argmin() # the index for the desired depth is found by locating the smallest value in the difference array
        sidx = sidx.astype(int)

        # Compute Love Numbers at Desired Depth: Sample the Y Solutions
        for kk in range(0,numrad):
            if (numrad == 1):
                csidx = sidx.copy()
            else:
                csidx = sidx[kk]
            # Compute Load Love Numbers
            hprime[kk],nlprime[kk],nkprime[kk] = evaluate_load_ln_n0.main(n,Y_load[csidx,:],adim,gsdim,T_sc,L_sc)
            # Compute Potential Love Numbers
            hpot[kk],nlpot[kk],nkpot[kk] = evaluate_potential_ln_n0.main(n)
            # Compute Stress Love Numbers
            hstr[kk],nlstr[kk],nkstr[kk] = evaluate_stress_ln_n0.main(n,Y_str[csidx,:],adim,gsdim,T_sc,L_sc)
            # Compute Shear Love Numbers: A Purely Shear Force Cannot Generate a Degree-0 Harmonic Response
            hshr[kk],nlshr[kk],nkshr[kk] = evaluate_shear_ln_n0.main(n)

    # Determine When to Integrate Through Cores, Versus When to Begin Integration in the Mantle (Note: r^n > inf_tol)
    # (r/a)^n > inf_tol : start integration at the r for which this expression holds
    # Logarithm power rule: log((r/a)^n) = n log(r/a) = log(inf_tol)
    # soc = roc/a, where roc = radius of the outer core
    # Thus: n = log(inf_tol)/log(soc) yields the n above which integrations should start in the mantle
    # To be safe, we round upward, and any n beneath that value will be integrated through the whole Earth
    # Note: If receiving an error like this: ValueError: operands could not be broadcast together with shapes (2,) (100,) 
    #       then consider further limiting the number of spherical-harmonic degrees integrated through full body
    # Mars AR Model from Okal & Anderson (1989), Table 4 --> nmaxfull = 8
    # ELLN model with fluid inner core --> nmaxfull = 8
    # Mars model from Khan et al. --> nmaxfull = 6
    elif ((n > 0) and (n < nmaxfull)):
 
        # Perform the Integration Through Full Earth
        Y1, Y2, Y3, sint_mt = integrate_fullEarth.main(n,s_min,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,sic,soc,small,\
            num_soln,backend,abs_tol,rel_tol,nstps,order,adim,gsdim,L_sc,T_sc,inf_tol,s,kx=kx)

        # Compute Solutions For Homogeneous Sphere (Size of Earth, for Different n)
        Y1_shape = np.asarray(Y1).shape
        Y = fundamental_solutions_homsph.main(1.,n,tck_lnd,tck_mnd,tck_rnd,wnd,piG)
        Y1i = Y[0,:]
        Y2i = Y[1,:]
        Y3i = Y[2,:]
        Y1 = np.ones((Y1_shape[0],Y1_shape[1]))*Y1i
        Y2 = np.ones((Y1_shape[0],Y1_shape[1]))*Y2i
        Y3 = np.ones((Y1_shape[0],Y1_shape[1]))*Y3i
        print(':: Reminder: Computing Y Solutions Based on Homogeneous Sphere.')

        # Apply Boundary Conditions at Surface
        m_load,m_pot,m_str,m_shr = apply_boundary_conditions.main(n,Y1[-1],Y2[-1],Y3[-1],gnd[-1],piG)

        # Compute Y Solutions
        Y_load = compute_solutions.main(Y1,Y2,Y3,m_load)
        Y_pot  = compute_solutions.main(Y1,Y2,Y3,m_pot)
        Y_str  = compute_solutions.main(Y1,Y2,Y3,m_str)
        Y_shr  = compute_solutions.main(Y1,Y2,Y3,m_shr)

        # Compute Love Numbers at Desired Depth: Find the appropriate radii
        if (numrad == 1):
            abs_val_diff = np.abs(np.asarray(sint_mt) - float(eval_radii)) # absolute values of the differences between the integration radii and the test depth
            sidx = abs_val_diff.argmin() # the index for the desired depth is found by locating the smallest value in the difference array
        else:
            for jj in range(0,numrad):
                abs_val_diff = np.abs(np.asarray(sint_mt) - float(eval_radii[jj])) # absolute values of the differences between the integration radii and the test depth
                sidx[jj] = abs_val_diff.argmin() # the index for the desired depth is found by locating the smallest value in the difference array
        sidx = sidx.astype(int)

        # Compute Love Numbers at Desired Depth: Sample the Y Solutions
        for kk in range(0,numrad):
            if (numrad == 1):
                csidx = sidx.copy()
                crad = eval_radii.copy()
            else:
                csidx = sidx[kk]
                crad = eval_radii[kk]
            # Compute Load Love Numbers
            hprime[kk],nlprime[kk],nkprime[kk] = evaluate_load_ln.main(n,Y_load[csidx,:],adim,gsdim,T_sc,L_sc)
            # Compute Potential Love Numbers
            hpot[kk],nlpot[kk],nkpot[kk] = evaluate_potential_ln.main(n,Y_pot[csidx,:],adim,gsdim,T_sc,L_sc)
            # Compute Stress Love Numbers
            hstr[kk],nlstr[kk],nkstr[kk] = evaluate_stress_ln.main(n,Y_str[csidx,:],adim,gsdim,T_sc,L_sc)
            # Compute Shear Love Numbers
            hshr[kk],nlshr[kk],nkshr[kk] = evaluate_shear_ln.main(n,Y_shr[csidx,:],adim,gsdim,T_sc,L_sc)
            # Check: If the Evaluation Radii are less than the Starting Radius of the Integration, then Set the Love Numbers at those Radii to NaN
            #print(sint_mt[csidx])
            if (float(crad) < min(sint_mt)):
                hprime[kk] = nlprime[kk] = nkprime[kk] = hpot[kk] = nlpot[kk] = nkpot[kk] = hstr[kk] = nlstr[kk] = nkstr[kk] = hshr[kk] = nlshr[kk] = nkshr[kk] = float("nan")

    else:

        # Perform the Integration Through the Mantle Only
        Y1, Y2, Y3, sint_mt = integrate_mantle.main(n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,\
            num_soln,backend,abs_tol,rel_tol,nstps,order,inf_tol,s,soc,kx=kx)

        # Compute Solutions For Homogeneous Sphere (Size of Earth, for Different n)
        Y1_shape = np.asarray(Y1).shape
        Y = fundamental_solutions_homsph.main(1.,n,tck_lnd,tck_mnd,tck_rnd,wnd,piG)
        Y1i = Y[0,:]
        Y2i = Y[1,:]
        Y3i = Y[2,:]
        Y1 = np.ones((Y1_shape[0],Y1_shape[1]))*Y1i
        Y2 = np.ones((Y1_shape[0],Y1_shape[1]))*Y2i
        Y3 = np.ones((Y1_shape[0],Y1_shape[1]))*Y3i
        print(':: Reminder: Computing Y Solutions Based on Homogeneous Sphere.')

        # Apply Boundary Conditions at Surface
        m_load,m_pot,m_str,m_shr = apply_boundary_conditions.main(n,Y1[-1],Y2[-1],Y3[-1],gnd[-1],piG)

        # Compute Y Solutions
        Y_load = compute_solutions.main(Y1,Y2,Y3,m_load)
        Y_pot  = compute_solutions.main(Y1,Y2,Y3,m_pot)
        Y_str  = compute_solutions.main(Y1,Y2,Y3,m_str)
        Y_shr  = compute_solutions.main(Y1,Y2,Y3,m_shr)

        # Compute Love Numbers at Desired Depth: Find the appropriate radii
        if (numrad == 1):
            abs_val_diff = np.abs(np.asarray(sint_mt) - float(eval_radii)) # absolute values of the differences between the integration radii and the test depth
            sidx = abs_val_diff.argmin() # the index for the desired depth is found by locating the smallest value in the difference array
        else: 
            for jj in range(0,numrad):
                abs_val_diff = np.abs(np.asarray(sint_mt) - float(eval_radii[jj])) # absolute values of the differences between the integration radii and the test depth
                sidx[jj] = abs_val_diff.argmin() # the index for the desired depth is found by locating the smallest value in the difference array
        sidx = sidx.astype(int)

        # Compute Love Numbers at Desired Depth: Sample the Y Solutions
        for kk in range(0,numrad):
            if (numrad == 1):
                csidx = sidx.copy()
                crad = eval_radii.copy()
            else:
                csidx = sidx[kk]
                crad = eval_radii[kk]
            # Compute Load Love Numbers
            hprime[kk],nlprime[kk],nkprime[kk] = evaluate_load_ln.main(n,Y_load[csidx,:],adim,gsdim,T_sc,L_sc)
            # Compute Potential Love Numbers
            hpot[kk],nlpot[kk],nkpot[kk] = evaluate_potential_ln.main(n,Y_pot[csidx,:],adim,gsdim,T_sc,L_sc)
            # Compute Stress Love Numbers
            hstr[kk],nlstr[kk],nkstr[kk] = evaluate_stress_ln.main(n,Y_str[csidx,:],adim,gsdim,T_sc,L_sc)
            # Compute Shear Love Numbers
            hshr[kk],nlshr[kk],nkshr[kk] = evaluate_shear_ln.main(n,Y_shr[csidx,:],adim,gsdim,T_sc,L_sc)
            # Check: If the Evaluation Radii are less than the Starting Radius of the Integration, then Set the Love Numbers at those Radii to NaN
            #print(sint_mt[csidx])
            if (float(crad) < min(sint_mt)):
                hprime[kk] = nlprime[kk] = nkprime[kk] = hpot[kk] = nlpot[kk] = nkpot[kk] = hstr[kk] = nlstr[kk] = nkstr[kk] = hshr[kk] = nlshr[kk] = nkshr[kk] = float("nan")

    # Flatten the Y Solution Vectors into 1d Arrays
    Y_load = Y_load.flatten()
    Y_pot  = Y_pot.flatten()
    Y_str  = Y_str.flatten()
    Y_shr  = Y_shr.flatten()
 
    # Return Love Numbers
    return hprime,nlprime,nkprime,hpot,nlpot,nkpot,hstr,nlstr,nkstr,hshr,nlshr,nkshr,sint_mt,Y_load,Y_pot,Y_str,Y_shr


