# *********************************************************************
# FUNCTION TO INTEGRATE Y-SOLUTIONS THROUGH THE FULL PLANET (n>=1)
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
from LOADGF.LN import fundamental_solutions_powser
from LOADGF.LN import fundamental_solutions_powser_Z
from LOADGF.LN import solid2fluid
from LOADGF.LN import fluid2solid
from LOADGF.LN import fundamental_solutions_homsph
from LOADGF.LN import integrate_f_fluid
from LOADGF.LN import integrate_f_solid
from LOADGF.LN import integrate_f_solid_Z
from scipy import interpolate

def main(n,s_min,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,sic,soc,small,\
    num_soln,backend,abs_tol,rel_tol,nstps,m,a,gs,L_sc,T_sc,inf_tol,s):
 
    # Compute Starting Solutions By Power Series Expansion
    #Y = fundamental_solutions_powser.main(s_min,n,tck_lnd,tck_mnd,tck_rnd,wnd,ond,piG,m); Z = None
    Z = fundamental_solutions_powser_Z.main(s_min,n,tck_lnd,tck_mnd,tck_rnd,wnd,ond,piG,m); Y = None
    # Compute Starting Solutions From Homogeneous Sphere
    #Y = fundamental_solutions_homsph.main(s_min,n,tck_lnd,tck_mnd,tck_rnd,wnd,piG); Z = None

    # Y Solutions
    if Y is not None:
        # Extract Y Solutions
        Y1i = Y[0,:]
        Y2i = Y[1,:]
        Y3i = Y[2,:]
        # Integrate Through Inner Core
        int_start = s_min
        int_stop  = sic-small
        Y1,sint1ic = integrate_f_solid.main(Y1i,int_start,int_stop,num_soln,backend,nstps,\
            abs_tol,rel_tol,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)
        Y2,sint2ic = integrate_f_solid.main(Y2i,int_start,int_stop,num_soln,backend,nstps,\
            abs_tol,rel_tol,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)
        Y3,sint3ic = integrate_f_solid.main(Y3i,int_start,int_stop,num_soln,backend,nstps,\
            abs_tol,rel_tol,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)
  
    # Z-transform Solutions (Smylie 2013)
    if Z is not None: 
        # Extract Z-transform Solutions
        Z1i       = Z[0,:]
        Z2i       = Z[1,:]
        Z3i       = Z[2,:]
        # Integrate Through Inner Core 
        int_start = s_min
        int_stop  = sic-small
        alpha     = n-2.
        Y1,sint1ic = integrate_f_solid_Z.main(Z1i,int_start,int_stop,num_soln,backend,nstps,\
            abs_tol,rel_tol,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m,alpha)
        Y2,sint2ic = integrate_f_solid_Z.main(Z2i,int_start,int_stop,num_soln,backend,nstps,\
            abs_tol,rel_tol,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m,alpha)
        alpha     = n 
        Y3,sint3ic = integrate_f_solid_Z.main(Z3i,int_start,int_stop,num_soln,backend,nstps,\
            abs_tol,rel_tol,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m,alpha)

    # Fluid Outer Core Exists
    if (sic < soc): 
 
        # Apply Boundary Conditions at Solid-Fluid Interface
        YOC1i, YOC2i = solid2fluid.main(Y1[-1],Y2[-1],Y3[-1])
 
        # Integrate Through Outer Core
        int_start = sic
        int_stop  = soc
        X1,sint1oc = integrate_f_fluid.main(YOC1i,int_start,int_stop,num_soln,backend,nstps,\
            abs_tol,rel_tol,n,tck_lnd,tck_rnd,tck_gnd,wnd,piG,m,ond)
        X2,sint2oc = integrate_f_fluid.main(YOC2i,int_start,int_stop,num_soln,backend,nstps,\
            abs_tol,rel_tol,n,tck_lnd,tck_rnd,tck_gnd,wnd,piG,m,ond)
 
        # Apply Boundary Conditions at Fluid-Solid Interface
        YMT1i, YMT2i, YMT3i = fluid2solid.main(X1[-1],X2[-1])
 
    # Fluid Outer Core Does Not Exist
    else:
    
        # Set Solutions at Top of Inner Core to Solutions at Base of "Mantle"
        YMT1i = Y1[-1]; YMT2i = Y2[-1]; YMT3i = Y3[-1]

    # Integrate Through Mantle
    int_start = soc+small
    int_stop  = s[-1]
    Y1,sint1mt = integrate_f_solid.main(YMT1i,int_start,int_stop,num_soln,backend,nstps,\
        abs_tol,rel_tol,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)
    Y2,sint2mt = integrate_f_solid.main(YMT2i,int_start,int_stop,num_soln,backend,nstps,\
        abs_tol,rel_tol,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)
    Y3,sint3mt = integrate_f_solid.main(YMT3i,int_start,int_stop,num_soln,backend,nstps,\
        abs_tol,rel_tol,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)

    # Return Mantle Solutions (sint = normalized radii at solution points)
    return Y1, Y2, Y3, sint1mt
    

