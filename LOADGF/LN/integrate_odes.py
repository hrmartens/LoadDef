# *********************************************************************
# FUNCTION TO INTEGRATE ODEs for SPHEROIDAL DEFORMATION OF AN ELASTIC BODY
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
from scipy import interpolate

def main(n,s_min,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,sic,soc,small,\
    num_soln,backend,abs_tol,rel_tol,nstps,order,gnd,adim,gsdim,L_sc,T_sc,inf_tol,s,nmaxfull):

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
  
    # Special Case: n=0
    if (n == 0):

        # Perform the Integration Through Full Earth
        Y1, Y2, sint_mt = integrate_fullEarth_n0.main(n,s_min,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,sic,soc,small,\
            num_soln,backend,abs_tol,rel_tol,nstps,order,adim,gsdim,L_sc,T_sc,inf_tol,s)
 
        # Apply Boundary Conditions at Surface
        m_load,m_pot,m_str,m_shr = apply_boundary_conditions_n0.main(n,Y1[-1],Y2[-1],gnd[-1],piG)

        # Compute Y Solutions
        Y_load = compute_solutions_n0.main(Y1,Y2,m_load)
        Y_pot  = compute_solutions_n0.main(Y1,Y2,m_pot)
        Y_str  = compute_solutions_n0.main(Y1,Y2,m_str)
        Y_shr  = compute_solutions_n0.main(Y1,Y2,m_shr)

        # Compute Load Love Numbers
        hprime,nlprime,nkprime = evaluate_load_ln_n0.main(n,Y_load[-1,:],adim,gsdim,T_sc,L_sc)

	# Potential Love Numbers
        hpot,nlpot,nkpot = evaluate_potential_ln_n0.main(n)

        # Stress Love Numbers
        hstr,nlstr,nkstr = evaluate_stress_ln_n0.main(n,Y_str[-1,:],adim,gsdim,T_sc,L_sc)

        # Shear Love Numbers: A Purely Shear Force Cannot Generate a Degree-0 Harmonic Response
        hshr,nlshr,nkshr = evaluate_shear_ln_n0.main(n)

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
            num_soln,backend,abs_tol,rel_tol,nstps,order,adim,gsdim,L_sc,T_sc,inf_tol,s)

        # Apply Boundary Conditions at Surface
        m_load,m_pot,m_str,m_shr = apply_boundary_conditions.main(n,Y1[-1],Y2[-1],Y3[-1],gnd[-1],piG)

        # Compute Y Solutions
        Y_load = compute_solutions.main(Y1,Y2,Y3,m_load)
        Y_pot  = compute_solutions.main(Y1,Y2,Y3,m_pot)
        Y_str  = compute_solutions.main(Y1,Y2,Y3,m_str)
        Y_shr  = compute_solutions.main(Y1,Y2,Y3,m_shr)

        # Compute Load Love Numbers
        hprime,nlprime,nkprime = evaluate_load_ln.main(n,Y_load[-1,:],adim,gsdim,T_sc,L_sc)

        # Compute Potential Love Numbers
        hpot,nlpot,nkpot = evaluate_potential_ln.main(n,Y_pot[-1,:],adim,gsdim,T_sc,L_sc)

        # Compute Stress Love Numbers
        hstr,nlstr,nkstr = evaluate_stress_ln.main(n,Y_str[-1,:],adim,gsdim,T_sc,L_sc)

        # Compute Shear Love Numbers
        hshr,nlshr,nkshr = evaluate_shear_ln.main(n,Y_shr[-1,:],adim,gsdim,T_sc,L_sc)

    else:

        # Perform the Integration Through the Mantle Only
        Y1, Y2, Y3, sint_mt = integrate_mantle.main(n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,\
            num_soln,backend,abs_tol,rel_tol,nstps,order,inf_tol,s,soc)

        # Apply Boundary Conditions at Surface
        m_load,m_pot,m_str,m_shr = apply_boundary_conditions.main(n,Y1[-1],Y2[-1],Y3[-1],gnd[-1],piG)

        # Compute Y Solutions
        Y_load = compute_solutions.main(Y1,Y2,Y3,m_load)
        Y_pot  = compute_solutions.main(Y1,Y2,Y3,m_pot)
        Y_str  = compute_solutions.main(Y1,Y2,Y3,m_str)
        Y_shr  = compute_solutions.main(Y1,Y2,Y3,m_shr)

        # Compute Load Love Numbers
        hprime,nlprime,nkprime = evaluate_load_ln.main(n,Y_load[-1,:],adim,gsdim,T_sc,L_sc)

        # Compute Potential Love Numbers
        hpot,nlpot,nkpot = evaluate_potential_ln.main(n,Y_pot[-1,:],adim,gsdim,T_sc,L_sc)

        # Compute Stress Love Numbers
        hstr,nlstr,nkstr = evaluate_stress_ln.main(n,Y_str[-1,:],adim,gsdim,T_sc,L_sc)

        # Compute Shear Love Numbers
        hshr,nlshr,nkshr = evaluate_shear_ln.main(n,Y_shr[-1,:],adim,gsdim,T_sc,L_sc)

    # Flatten the Y Solution Vectors into 1d Arrays
    Y_load = Y_load.flatten()
    Y_pot  = Y_pot.flatten()
    Y_str  = Y_str.flatten()
    Y_shr  = Y_shr.flatten()

    # Return Love Numbers
    return hprime,nlprime,nkprime,hpot,nlpot,nkpot,hstr,nlstr,nkstr,hshr,nlshr,nkshr,sint_mt,Y_load,Y_pot,Y_str,Y_shr


