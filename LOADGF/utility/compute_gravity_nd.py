# *********************************************************************
# FUNCTION TO COMPUTE GRAVITY BASED ON PLANETARY MASS DISTRIBUTION
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

def main(s,rnd,piG,L_sc,T_sc):

    # Sort Arrays by Radius
    sorts = np.argsort(s)
    sorteds = s.copy()
    sorteds = s[sorts]
    sortedrnd = rnd.copy()
    sortedrnd = rnd[sorts]

    gnd = np.zeros(len(s))
    mass_enclosed = np.zeros(len(s))
    # NOTE that piG = 1 (therefore do not need to write explicitly in equations below)
    for jj in range(1,len(s)):
        volume_shell = (4./3)*(sorteds[jj]**3) - (4./3)*(sorteds[jj-1]**3)
        mass_enclosed[jj] = (volume_shell*(sortedrnd[jj]+sortedrnd[jj-1])/2.) + \
            mass_enclosed[jj-1]
        gnd[jj] = (mass_enclosed[jj]/(sorteds[jj]**2))

    # Convert to dimensionalized gravity
    g = gnd * (L_sc*(T_sc**(-2)))

    return g,gnd

