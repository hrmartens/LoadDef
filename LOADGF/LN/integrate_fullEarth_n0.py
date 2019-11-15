# *********************************************************************
# FUNCTION TO INTEGRATE Y-SOLUTIONS THROUGH THE FULL PLANET (n=0)
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
from LOADGF.LN import fundamental_solutions_powser_n0
from LOADGF.LN import fundamental_solutions_powser_Z_n0
from LOADGF.LN import integrate_f_fluid_n0
from LOADGF.LN import integrate_f_solid_n0
from LOADGF.LN import integrate_f_solid_Z_n0
from scipy import interpolate

def main(n,s_min,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,sic,soc,small,\
    num_soln,backend,abs_tol,rel_tol,nstps,m,a,gs,L_sc,T_sc,inf_tol,s):
 
    # Special Case: n=0
    if (n == 0):

        # Compute Starting Solutions by Power Series Expansion (Z-transform Solutions)
        Z = fundamental_solutions_powser_Z_n0.main(s_min,tck_lnd,tck_mnd,tck_rnd,wnd,ond,piG); Y = None
        
        # Compute Starting Solutions From Power Series Expansion (Y Solutions)
        #Y = fundamental_solutions_powser_n0.main(s_min,tck_lnd,tck_mnd,tck_rnd,wnd,ond,piG); Z = None

        # Y Solutions 
        if Y is not None:
            # Extract Y Solutions
            Y1i = Y[0,:]
            Y2i = Y[1,:]
 
            # Integrate Through Inner Core
            int_start = s_min
            int_stop  = sic-small
            Y1,sint1ic = integrate_f_solid_n0.main(Y1i,int_start,int_stop,num_soln,backend,nstps,\
                abs_tol,rel_tol,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG)
            Y2,sint2ic = integrate_f_solid_n0.main(Y2i,int_start,int_stop,num_soln,backend,nstps,\
                abs_tol,rel_tol,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG)

        # Z-transform Solutions
        if Z is not None:
            # Extract Z-transform Solutions
            Z1i       = Z[0,:]
            Z2i       = Z[1,:]
            # Integrate Through Inner Core
            alpha     = 0.
            int_start = s_min
            int_stop  = sic-small
            Y1,sint1ic = integrate_f_solid_Z_n0.main(Z1i,int_start,int_stop,num_soln,backend,nstps,\
                abs_tol,rel_tol,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,alpha)
            Y2,sint2ic = integrate_f_solid_Z_n0.main(Z2i,int_start,int_stop,num_soln,backend,nstps,\
                abs_tol,rel_tol,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,alpha)

        # Fluid Outer Core Exists
        if (sic < soc):

            # Apply Boundary Conditions at Solid-Fluid Interface
            YOC1i = Y1[-1]
            YOC2i = Y2[-1]

            # Integrate Through Outer Core
            int_start = sic
            int_stop  = soc
            X1,sint1oc = integrate_f_fluid_n0.main(YOC1i,int_start,int_stop,num_soln,backend,nstps,\
                abs_tol,rel_tol,tck_lnd,tck_rnd,tck_gnd,wnd,ond,piG)
            X2,sint2oc = integrate_f_fluid_n0.main(YOC2i,int_start,int_stop,num_soln,backend,nstps,\
                abs_tol,rel_tol,tck_lnd,tck_rnd,tck_gnd,wnd,ond,piG)

            # Apply Boundary Conditions at Fluid-Solid Interface
            YMT1i = X1[-1]
            YMT2i = X2[-1]

        # Fluid Outer Core Does Not Exist
        else:

            # Solutions at Top of Inner Core are Set Equivalent to Base of "Mantle"
            YMT1i = Y1[-1]; YMT2i = Y2[-1]

        # Integrate Through Mantle
        int_start = soc+small
        int_stop  = s[-1]
        Y1,sint1mt = integrate_f_solid_n0.main(YMT1i,int_start,int_stop,num_soln,backend,nstps,\
            abs_tol,rel_tol,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG)
        Y2,sint2mt = integrate_f_solid_n0.main(YMT2i,int_start,int_stop,num_soln,backend,nstps,\
            abs_tol,rel_tol,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG)

    else:
         
        # If n=!1, then Exit with an Error Message
        sys.exit('Error: n must = 0 in integrate_fullEarth_n0.py')

    # Return Mantle Solutions (sint = normalized radii at solution points)
    return Y1, Y2, sint1mt


