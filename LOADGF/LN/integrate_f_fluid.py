# *********************************************************************
# FUNCTION TO INTEGRATE Y-SOLUTIONS THROUGH FLUID LAYER (n>=1)
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
from scipy.integrate import ode
from LOADGF.LN import f_fluid

def main(Yi,int_start,int_stop,num_soln,backend,nstps,\
    abs_tol,rel_tol,n,tck_lnd,tck_rnd,tck_gnd,wnd,piG,m,ond):

    # Compute Solution Step Size
    dsc = (int_stop-int_start)/num_soln
    eps = dsc/10.

    # Initialize Solution Arrays
    Yint = []
    sint = []

    # Initialize Solver
    solver = ode(f_fluid.main).set_integrator(backend,atol=abs_tol,rtol=rel_tol,nsteps=nstps)
    solver.set_initial_value(Yi,int_start).set_f_params(n,tck_lnd,tck_rnd,tck_gnd,wnd,piG,m,ond)

    # Integrate
    while solver.successful() and (solver.t + dsc) < int_stop-eps:
        solver.integrate(solver.t + dsc)
        Yint.append(solver.y)
        sint.append(solver.t)

    # If not yet to the stopping radius, continue one last step 
    if (max(sint) < int_stop):
        solver.integrate(int_stop)
        Yint.append(solver.y)
        sint.append(solver.t)

    # Return Solution Vector
    return Yint,sint

