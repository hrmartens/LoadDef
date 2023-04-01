# *********************************************************************
# FUNCTION TO COMPUTE STRESS LOVE NUMBERS FROM Y-SOLUTIONS (n=0)
# Stress solutions only apply to n=1, but evaluated here for consistency across code
# See Okubo & Endo (1986) Geophys. J. R. astr. Soc. 
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

def main(n,Y_str,a,gs,T_sc,L_sc):

    # Extract Solutions
    Y1sol_str = Y_str[0]
    Y2sol_str = Y_str[1]
    Y3sol_str = Y_str[2]
    Y4sol_str = Y_str[3]

    # Block of code here has same effect as n=1 adjustment to Love numbers below. Only need one of the two blocks!
    #if (n == 1): 
    #    # Add a Rigid Body Rotation (Merriam, 1985)
    #    #  Equivalent to Takeuchi & Saito (1972) Eq. 100 for n=1
    #    gsnd = (gs/(L_sc*(T_sc**(-2))))
    #    alpha = -Y5sol_str/gsnd # Define Alpha Such That Y5 = 0
    #    Y1sol_str += (1.*alpha)
    #    Y2sol_str += 0.
    #    Y3sol_str += (1.*alpha)
    #    Y4sol_str += 0.
    #    Y5sol_str += (gsnd*alpha)
    #    Y6sol_str += (-2.*gsnd*alpha)

    # Compute Stress Love Numbers
    hstr = Y1sol_str
    nlstr = n*Y3sol_str

    # Adjust degree-one Love numbers to ensure that the potential field 
    # outside the Earth vanishes in the CE frame (e.g. Merriam 1985)
    # DO NOT DO THIS FOR CASE OF NO GRAVITY (for no gravity, n=1 is not well defined; no constraint)
    #if (n == 1):
    #    hstr = hstr - nkstr
    #    nlstr = nlstr - nkstr
    #    nkstr = nkstr - nkstr

    # Flatten Arrays
    hstr    = hstr.flatten()
    nlstr   = nlstr.flatten()
    nkstr   = np.multiply(nlstr,0.)

    # Return Variables
    return hstr,nlstr,nkstr


