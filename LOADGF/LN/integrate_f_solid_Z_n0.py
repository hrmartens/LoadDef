# *********************************************************************
# FUNCTION TO INTEGRATE TRANSFORMED Y-SOLUTIONS THROUGH A SOLID LAYER (n=0)
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

from __future__ import print_function
import numpy as np
from scipy.integrate import ode
from LOADGF.LN import f_solid_Z_n0
from LOADGF.LN import Z2Y_n0

def main(Zi,int_start,int_stop,num_soln,backend,nstps,abs_tol,rel_tol,\
    tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,alpha):

    # Compute Solution Step Size
    dsc = (int_stop - int_start)/num_soln
    eps = dsc/10.

    # Initialize Solution Arrays
    Zint = []
    sint = []
 
    # Initialize Solver
    solver = ode(f_solid_Z_n0.main).set_integrator(backend,atol=abs_tol,rtol=rel_tol,nsteps=nstps)
    solver.set_initial_value(Zi,int_start).set_f_params(tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,alpha)
    
    # Integrate
    while solver.successful() and (solver.t + dsc) < int_stop-eps:
        solver.integrate(solver.t + dsc)
        Zint.append(solver.y)
        sint.append(solver.t)

    # If not yet to the stopping radius, continue one last step
    if (max(sint) < int_stop):
        solver.integrate(int_stop)
        Zint.append(solver.y)
        sint.append(solver.t)

    # Convert Back to Y-Variables (Smylie Convention)
    Yint = list(Zint) # Copy the list into a new list Yint
    for oo in range(0,len(sint)):
        Yint[oo] = Z2Y_n0.main(sint[oo],alpha,Zint[oo])

    # Return Solution Vector
    return Yint,sint

