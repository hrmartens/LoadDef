# *********************************************************************
# FUNCTION TO COMPUTE STRESS LOVE NUMBERS FROM Y-SOLUTIONS (n=0)
# Stress solutions only apply to n=1, but evaluated here for consistency across code
# See Okubo & Endo (1986) Geophys. J. R. astr. Soc. 
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

def main(n,Y_str,a,gs,T_sc,L_sc):

    # Extract Solutions
    Y1sol_str = Y_str[0]
    Y2sol_str = Y_str[1]
    Y5sol_str = Y_str[2]
    Y6sol_str = Y_str[3]

    # Compute Stress Love Numbers
    hstr = Y1sol_str
    nlstr = np.array([[0.]])
    Y5sol = Y5sol_str*((L_sc**2.)*(T_sc**(-2.)))
    nkstr = np.array([[0.]])

    # Flatten Arrays
    hstr    = hstr.flatten()
    nlstr   = nlstr.flatten()
    nkstr   = nkstr.flatten()

    # Return Variables
    return hstr,nlstr,nkstr


