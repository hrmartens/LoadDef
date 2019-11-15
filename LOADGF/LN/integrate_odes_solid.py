# *********************************************************************
# FUNCTION TO INTEGRATE ODEs FOR SPHEROIDAL DEFORMATION THROUGH A SOLID BODY
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
from LOADGF.LN import integrate_fullEarth_solid_n0
from LOADGF.LN import integrate_fullEarth_solid
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
from LOADGF.LN import compute_Y_solutions_n0
from LOADGF.LN import compute_Y_solutions
from scipy import interpolate

def main(n,s_min,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,sic,soc,small,\
    num_soln,backend,abs_tol,rel_tol,nstps,m,gnd,a,gs,L_sc,T_sc,inf_tol,s,tolfac):

    # Special Case: n=0
    if (n == 0):

        # Perform the Integration Through Full [SOLID] Earth
        Y1, Y2, sint_mt = integrate_fullEarth_solid_n0.main(n,s_min,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,sic,soc,small,\
            num_soln,backend,abs_tol,rel_tol,nstps,m,gnd,a,gs,L_sc,T_sc,inf_tol,s,tolfac)
 
        # Apply Boundary Conditions at Surface
        m_load,m_pot,m_str,m_shr = apply_boundary_conditions_n0.main(n,Y1[-1],Y2[-1],gnd[-1],piG)

        # Compute Y Solutions
        Y_load = compute_Y_solutions_n0.main(Y1,Y2,m_load)
        Y_pot  = compute_Y_solutions_n0.main(Y1,Y2,m_pot)
        Y_str  = compute_Y_solutions_n0.main(Y1,Y2,m_str)
        Y_shr  = compute_Y_solutions_n0.main(Y1,Y2,m_shr)

        # Compute Load Love Numbers
        hprime,nlprime,nkprime = evaluate_load_ln_n0.main(n,Y_load[-1,:],a,gs,T_sc,L_sc)

        # Potential Love Numbers
        hpot,nlpot,nkpot = evaluate_potential_ln_n0.main(n)

        # Stress Love Numbers
        hstr,nlstr,nkstr = evaluate_stress_ln_n0.main(n,Y_str[-1,:],a,gs,T_sc,L_sc)

        # Shear Love Numbers: A Purely Shear Force Cannot Generate a Degree-0 Harmonic Response
        hshr,nlshr,nkshr = evaluate_shear_ln_n0.main(n)
    
    elif ((n > 0) and (n < (math.ceil(math.log(inf_tol)/math.log(soc))))): 
      
        # Perform the Integration Through Full [SOLID] Earth
        Y1, Y2, Y3, sint_mt = integrate_fullEarth_solid.main(n,s_min,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,sic,soc,small,\
            num_soln,backend,abs_tol,rel_tol,nstps,m,gnd,a,gs,L_sc,T_sc,inf_tol,s,tolfac)      
  
        # Apply Boundary Conditions at Surface
        m_load,m_pot,m_str,m_shr = apply_boundary_conditions.main(n,Y1[-1],Y2[-1],Y3[-1],gnd[-1],piG)

        # Compute Y Solutions
        Y_load = compute_Y_solutions.main(Y1,Y2,Y3,m_load)
        Y_pot  = compute_Y_solutions.main(Y1,Y2,Y3,m_pot)
        Y_str  = compute_Y_solutions.main(Y1,Y2,Y3,m_str)
        Y_shr  = compute_Y_solutions.main(Y1,Y2,Y3,m_shr)

        # Compute Load Love Numbers
        hprime,nlprime,nkprime = evaluate_load_ln.main(n,Y_load[-1,:],a,gs,T_sc,L_sc)

        # Compute Potential Love Numbers
        hpot,nlpot,nkpot = evaluate_potential_ln.main(n,Y_pot[-1,:],a,gs,T_sc,L_sc)

        # Compute Stress Love Numbers
        hstr,nlstr,nkstr = evaluate_stress_ln.main(n,Y_str[-1,:],a,gs,T_sc,L_sc)

        # Compute Shear Love Numbers
        hshr,nlshr,nkshr = evaluate_shear_ln.main(n,Y_shr[-1,:],a,gs,T_sc,L_sc)
 
    else:

        # Perform the Integration Through the Mantle Only
        Y1, Y2, Y3, sint_mt = integrate_mantle.main(n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,\
            num_soln,backend,abs_tol,rel_tol,nstps,m,inf_tol,s)

        # Apply Boundary Conditions at Surface
        m_load,m_pot,m_str,m_shr = apply_boundary_conditions.main(n,Y1[-1],Y2[-1],Y3[-1],gnd[-1],piG)

        # Compute Y Solutions
        Y_load = compute_Y_solutions.main(Y1,Y2,Y3,m_load)
        Y_pot  = compute_Y_solutions.main(Y1,Y2,Y3,m_pot)
        Y_str  = compute_Y_solutions.main(Y1,Y2,Y3,m_str)
        Y_shr  = compute_Y_solutions.main(Y1,Y2,Y3,m_shr)

        # Compute Load Love Numbers
        hprime,nlprime,nkprime = evaluate_load_ln.main(n,Y_load[-1,:],a,gs,T_sc,L_sc)

        # Compute Potential Love Numbers
        hpot,nlpot,nkpot = evaluate_potential_ln.main(n,Y_pot[-1,:],a,gs,T_sc,L_sc)

        # Compute Stress Love Numbers
        hstr,nlstr,nkstr = evaluate_stress_ln.main(n,Y_str[-1,:],a,gs,T_sc,L_sc)

        # Compute Shear Love Numbers
        hshr,nlshr,nkshr = evaluate_shear_ln.main(n,Y_shr[-1,:],a,gs,T_sc,L_sc)

    # Flatten the Y Solution Vectors into 1d Arrays
    Y_load = Y_load.flatten()
    Y_pot  = Y_pot.flatten()
    Y_str  = Y_str.flatten()
    Y_shr  = Y_shr.flatten()

    # Return Variables
    return hprime,nlprime,nkprime,hpot,nlpot,nkpot,hstr,nlstr,nkstr,hshr,nlshr,nkshr,sint_mt,Y_load,Y_pot,Y_str,Y_shr


