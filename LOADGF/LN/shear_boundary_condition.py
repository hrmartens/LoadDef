# *********************************************************************
# FUNCTION TO APPLY BOUNDARY CONDITIONS FOR SURFACE SHEAR FORCING
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

def main(n,Y1s,Y2s,Y3s,gs_nondim,piG):

    # See Okubo & Saito (1983), Saito (1978)

    if (n == 0):

        # Solutions are Undefined for n=0; Simply Set to Zero
        Y1sol = Y2sol = Y3sol = Y4sol = Y5sol = Y6sol = np.array([[0.]])

        # Also Set mvec to zero
        mvec = [0.,0.,0.]

    else: 

        # Form Data Matrix Describing Load
        dmat = np.array([[0., (((2.*n+1.)*(gs_nondim**2))/((4.*piG*n)*(n+1))), 0.]])

        # Form G Matrix From Integrated Solutions
        Gmat = np.array([[Y1s[1], Y2s[1], Y3s[1]], \
            [Y1s[3], Y2s[3], Y3s[3]], \
            [(Y1s[5]+(n+1.)*Y1s[4]), (Y2s[5]+(n+1.)*Y2s[4]), (Y3s[5]+(n+1.)*Y3s[4])]])

        # Solve the System of Equations (NO Matrix Inversion, for Stability) 
        mvec = np.linalg.solve(Gmat,dmat.T) 
  
        # Compute Solutions
        Y1sol = np.dot(np.array([[Y1s[0], Y2s[0], Y3s[0]]]),mvec)
        Y2sol = np.dot(np.array([[Y1s[1], Y2s[1], Y3s[1]]]),mvec)
        Y3sol = np.dot(np.array([[Y1s[2], Y2s[2], Y3s[2]]]),mvec)
        Y4sol = np.dot(np.array([[Y1s[3], Y2s[3], Y3s[3]]]),mvec)
        Y5sol = np.dot(np.array([[Y1s[4], Y2s[4], Y3s[4]]]),mvec)
        Y6sol = np.dot(np.array([[Y1s[5], Y2s[5], Y3s[5]]]),mvec)
 
    # Return Solutions
    return Y1sol, Y2sol, Y3sol, Y4sol, Y5sol, Y6sol, mvec

