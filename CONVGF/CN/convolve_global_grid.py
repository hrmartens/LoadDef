# *********************************************************************
# FUNCTION TO CONVOLVE INTEGRATED GREENS FUNCTIONS WITH LOAD MODEL
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

def main(c1,c2,gfr,gfe,gfn):

    # Perform Convolution
    ec1 = np.multiply(c1,gfe)
    ec2 = np.multiply(c2,gfe)
    nc1 = np.multiply(c1,gfn)
    nc2 = np.multiply(c2,gfn)
    vc1 = np.multiply(c1,gfr)
    vc2 = np.multiply(c2,gfr)

    # Return Harmonic Coefficients for All 3 Components
    return ec1,ec2,nc1,nc2,vc1,vc2

