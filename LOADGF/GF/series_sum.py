# *********************************************************************
# FUNCTION TO COMPUTE RECURSIVE AVERAGES TO FACILITATE CONVERGENCE
# See Guo et al. (2004), EQ.S 23 & 24
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

def main(series):

    # Initialize Counter
    ii = 0
  
    while (ii < 100):
	
        # Initialize Array and Counter
        newterm = np.zeros(len(series))
        cc = 0

        for kk in range(0,len(series)):

            # Current Value
            cval = series[kk]
	
            if (kk == 0):
                # Half the First Value
                newterm[cc] = 0.5*cval
            else:
                pval = series[kk-1]
                # Average Current and Previous Values
                newterm[cc] = 0.5*(cval+pval)

            # Update Counter
            cc = cc + 1

        # Half the Last Value (Do NOT want to do this: we are truncating the series, and tapering)
        # newterm = np.append(newterm,[0.5*cval])        
 
        # Averaged Series Becomes New Series
        series = newterm.copy()

        # Update Counter
        ii = ii + 1

    # Compute Sum of Resulting Series
    summed = np.sum(series)

    # Return Summed Array
    return summed


