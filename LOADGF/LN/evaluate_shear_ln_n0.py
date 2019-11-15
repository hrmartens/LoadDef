# *********************************************************************
# FUNCTION TO COMPUTE SHEAR LOVE NUMBERS FROM Y-SOLUTIONS (n=0)
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

def main(n):

    # Shear Love Numbers: A Purely Shear Force Cannot Generate a Degree-0 Harmonic Response
    #  Degree-0 Displacements are Purely Radial (e.g., Okubo 1993, Sec. 4): The BCs Become Singular
    hshr = np.array([[0.]]) 
    nlshr = np.array([[0.]])
    nkshr = np.array([[0.]])
 
    # Flatten Arrays
    hshr    = hshr.flatten()
    nlshr   = nlshr.flatten()
    nkshr   = nkshr.flatten()

    # Return Variables
    return hshr,nlshr,nkshr


