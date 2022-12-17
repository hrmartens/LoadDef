# *********************************************************************
# FUNCTION TO APPLY BOUNDARY CONDITIONS FOR SURFACE MASS LOADING (n=0)
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
import sys

def main(n,Y1s,gs_nondim,piG):

    # Apply Boundary Conditions at Surface
    dmat = np.array([[-(gs_nondim**2)/(4.*piG)]])

    # Form G Matrix
    Gmat = np.array([[Y1s[1]]])

    # Solve the System of Equations (NO Matrix Inversion, for Stability) 
    mvec = np.linalg.solve(Gmat,dmat.T) 

    # Compute Solutions
    Y1sol = np.dot(np.array([[Y1s[0]]]),mvec)
    Y2sol = np.dot(np.array([[Y1s[1]]]),mvec)
    
    # Return Solutions
    return Y1sol, Y2sol, mvec

