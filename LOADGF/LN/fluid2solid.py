# *********************************************************************
# FUNCTION TO PROPAGATE Y-SOLUTIONS ACROSS A FLUID-SOLID INTERFACE
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

# Boundary Conditions for a Fluid --> Solid Interface
# See Takeuchi & Saito (1972), PG. 254

import numpy as np

def main(Y1,Y2):

    # Initialize Arrays
    Y1s = np.zeros(6)
    Y2s = np.zeros(6)
    Y3s = np.zeros(6)

    # Compute Three Solutions for Solid Layer
    # from Two Solutions for Fluid Layer
    Y1s[0] = Y1[0]
    Y1s[1] = Y1[1]
    Y1s[4] = Y1[2]
    Y1s[5] = Y1[3]

    Y2s[0] = Y2[0]
    Y2s[1] = Y2[1]
    Y2s[4] = Y2[2]
    Y2s[5] = Y2[3]

    Y3s[2] = 1.

    # Return Three Solutions for Solid Layer
    return Y1s, Y2s, Y3s

