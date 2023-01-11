# *********************************************************************
# FUNCTION TO ENFORCE MASS CONSERVATION
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

def main(ic1,ic2,iarea):

    # See Agnew (1983), Geophys. J. R. astr. Soc. 
    # Note that it does not matter if "iarea" is the unit area of the 
    #  spherical patch or if it's been multiplied by R^2. 
    #  "R^2" appears in every term and therefore cancels out. 
    ic1 = ic1 - np.divide(np.sum(np.multiply(ic1,iarea)),\
        (np.sum(iarea)))
    ic2 = ic2 - np.divide(np.sum(np.multiply(ic2,iarea)),\
        (np.sum(iarea)))

    # Return Variables
    return ic1,ic2

