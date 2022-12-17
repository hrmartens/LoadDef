# *********************************************************************
# FUNCTION TO COMPUTE SOLUTION VECTORS for n>=1
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

def main(Y1,Y2,mvec):

    # Convert Solution Lists to Arrays
    Y1_arr = np.array(Y1)
    Y2_arr = np.array(Y2)

    # Extract the Individual y's from the Arrays
    if (len(Y1_arr) == 4): # Only 1 Solution
        Y1_0 = Y1_arr[0]; Y1_1 = Y1_arr[1]; Y1_2 = Y1_arr[2]; Y1_3 = Y1_arr[3]
        Y2_0 = Y2_arr[0]; Y2_1 = Y2_arr[1]; Y2_2 = Y2_arr[2]; Y2_3 = Y2_arr[3]
    else:
        Y1_0 = Y1_arr[:,0]; Y1_1 = Y1_arr[:,1]; Y1_2 = Y1_arr[:,2]; Y1_3 = Y1_arr[:,3]
        Y2_0 = Y2_arr[:,0]; Y2_1 = Y2_arr[:,1]; Y2_2 = Y2_arr[:,2]; Y2_3 = Y2_arr[:,3]

    # Form Linear Combinations of the y's for the two Solution Vectors
    sol1 = np.array([Y1_0, Y2_0]).T
    sol2 = np.array([Y1_1, Y2_1]).T
    sol3 = np.array([Y1_2, Y2_2]).T
    sol4 = np.array([Y1_3, Y2_3]).T

    # Compute Solutions (e.g. Y1sol = m1*Y1_0 + m2*Y2_0)
    Y1sol = np.dot(sol1,mvec)
    Y2sol = np.dot(sol2,mvec)
    Y3sol = np.dot(sol3,mvec)
    Y4sol = np.dot(sol4,mvec)

    # Combine Solutions into Single Array
    Ysol = np.column_stack((Y1sol,Y2sol,Y3sol,Y4sol))

    # Return Solutions
    return Ysol

