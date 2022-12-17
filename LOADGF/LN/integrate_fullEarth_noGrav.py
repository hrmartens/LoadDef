# *********************************************************************
# FUNCTION TO INTEGRATE Y-SOLUTIONS THROUGH THE FULL PLANET (n>=1)
# SPECIAL CASE FOR NO GRAVITY
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
from LOADGF.LN import fundamental_solutions_homsph_noGrav
from LOADGF.LN import integrate_f_solid_noGrav
from scipy import interpolate

def main(n,s_min,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,sic,soc,small,\
    num_soln,backend,abs_tol,rel_tol,nstps,m,a,gs,L_sc,T_sc,inf_tol,s,kx=1):

    # !! Not appropriate/possible to integrate through liquid outer core in case of no self-gravity !!
    # The transition from fluid core to solid mantle yields solution vectors with zeros in multiple
    # fields and it is not possible to continue integrating to the surface.
    # As per conversation with Martin van Driel (July 2018) and reference to paper(s) by David Al Attar,
    # it is not appropriate to include the core regions when excluding gravity from the problem.
    # Hence, I will start integration from the CMB.

    # Compute Starting Solutions From Homogeneous Sphere
    Y = fundamental_solutions_homsph_noGrav.main(s_min,n,tck_lnd,tck_mnd,tck_rnd,wnd,piG)

    # Extract Y Solutions
    Y1i = Y[0,:]
    Y2i = Y[1,:]
    # Integrate Through Inner Core
    int_start = s_min
    int_stop  = sic-small 
    # integrate the 2 solutions at once, using 12 degrees of freedom in the ODE
    Y12i = np.concatenate([Y1i, Y2i])
    Y12, sint1ic = integrate_f_solid_noGrav.main(Y12i,int_start,int_stop,num_soln,backend,nstps,\
        abs_tol,rel_tol,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m,kx=kx)
    # unpack the 2 solutions
    Y12 = np.array(Y12)
    Y1 = list(Y12[:, 0:4])
    Y2 = list(Y12[:, 4:8])

    # Fluid Outer Core Exists
    if (sic < soc): 

        # !! Not appropriate/possible to integrate through liquid outer core in case of no self-gravity !!
        # The transition from fluid core to solid mantle yields solution vectors with zeros in multiple
        # fields and it is not possible to continue integrating to the surface.
        # As per conversation with Martin van Driel (July 2018) and reference to paper(s) by David Al Attar,
        # it is not appropriate to include the core regions when excluding gravity from the problem.
        # Hence, I will start integration from the CMB.

        # Compute Starting Solutions From Homogeneous Sphere
        print('WARNING: Since self-gravity is not included, it is inappropriate to consider the fluid outer core. Solutions for low spherical-harmonic degrees will commence at the CMB!!')
        Y = fundamental_solutions_homsph_noGrav.main((soc+small),n,tck_lnd,tck_mnd,tck_rnd,wnd,piG)
        YMT1i = Y[0,:]
        YMT2i = Y[1,:] 
 
    # Fluid Outer Core Does Not Exist
    else:

        # Set Solutions at Top of Inner Core to Solutions at Base of "Mantle"
        YMT1i = Y1[-1]; YMT2i = Y2[-1]

    # Integrate Through Mantle
    int_start = soc+small
    int_stop  = s[-1]
    Y12i = np.concatenate([YMT1i, YMT2i])
    Y12, sint1mt = integrate_f_solid_noGrav.main(Y12i,int_start,int_stop,num_soln,backend,nstps,\
        abs_tol,rel_tol,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m,kx=kx)
    # unpack the 2 solutions
    Y12 = np.array(Y12)
    Y1 = list(Y12[:, 0:4])
    Y2 = list(Y12[:, 4:8])

    # Return Mantle Solutions (sint = normalized radii at solution points)
    return Y1, Y2, sint1mt

