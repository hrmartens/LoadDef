# *********************************************************************
# FUNCTION TO INTEGRATE Y-SOLUTIONS THROUGH A MANTLE LAYER
# SPECIAL CASE OF NO GRAVITY
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

def main(n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,\
    num_soln,backend,abs_tol,rel_tol,nstps,m,inf_tol,s,soc,**kwargs):

    # Determine Starting Radius
    s_start_ind = np.where((s**n) > inf_tol)
    s_start_ind = s_start_ind[0]
    s_start = s[s_start_ind[0]]
    # Do not start in fluid (assumes fluid is not up close to surface)
    if (s_start <= soc):
        print(':: Warning: Forcing integration to start from top of fluid layer [integrate_mantle.py]')
        s_start = soc+0.001

    # Unpack any kwargs
    if 's_0' in kwargs:
        s_start = kwargs['s_0']
    if 'kx' in kwargs:
        kx = kwargs['kx']

    # ***************** HOMOGENEOUS SPHERE W/ Y-VARIABLES ********************* #
    # Compute Starting Solutions From Homogeneous Sphere
    Y = fundamental_solutions_homsph_noGrav.main(s_start,n,tck_lnd,tck_mnd,tck_rnd,wnd,piG)
    Y1i = Y[0,:]
    Y2i = Y[1,:]
    # Integrate From Starting Radius to Surface
    int_start = s_start
    int_stop  = s[-1]

    # integrate the 2 solutions at once, using 12 degrees of freedom in the ODE
    Y12i = np.concatenate([Y1i, Y2i])
    Y12, sint1mt = integrate_f_solid_noGrav.main(Y12i,int_start,int_stop,num_soln,backend,nstps,\
        abs_tol,rel_tol,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m,kx=kx)

    # unpack the 2 solutions
    Y12 = np.array(Y12)
    Y1 = list(Y12[:, 0:4])
    Y2 = list(Y12[:, 4:8])
    # ************************************************************************* #

    # Return Solutions (sint = normalized radii at solution locations)
    return Y1, Y2, sint1mt


