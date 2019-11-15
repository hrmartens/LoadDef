# *********************************************************************
# FUNCTION TO PROPAGATE Y-SOLUTIONS ACROSS A SOLID-FLUID INTERFACE
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

# Boundary Conditions for a Solid --> Fluid Interface
# See Takeuchi & Saito (1972), PG. 254

import numpy as np

def main(Y1,Y2,Y3):

    # Initialize Arrays
    Y1f = np.zeros(4)
    Y2f = np.zeros(4)

    # Compute Two Solutions for Fluid Layer 
    # from Three Solutions for Solid Layer
    Y1f[0] = Y1[0] - (Y1[3]/Y3[3])*Y3[0]
    Y1f[1] = Y1[1] - (Y1[3]/Y3[3])*Y3[1]
    Y1f[2] = Y1[4] - (Y1[3]/Y3[3])*Y3[4]
    Y1f[3] = Y1[5] - (Y1[3]/Y3[3])*Y3[5]

    Y2f[0] = Y2[0] - (Y2[3]/Y3[3])*Y3[0]
    Y2f[1] = Y2[1] - (Y2[3]/Y3[3])*Y3[1]
    Y2f[2] = Y2[4] - (Y2[3]/Y3[3])*Y3[4]
    Y2f[3] = Y2[5] - (Y2[3]/Y3[3])*Y3[5]

    # Return Two Solutions for Fluid Layer
    return Y1f, Y2f    

