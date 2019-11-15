# *********************************************************************
# FUNCTION TO GENERATE INTERPOLATION FUNCTIONS FOR MATERIAL PARAMETERS
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

from scipy import interpolate

def main(s,lnd,mnd,rnd,gnd,kx=1):

    # Interpolate Parameters | k=1 (Linear)
    tck_lnd = interpolate.splrep(s,lnd,k=kx)
    tck_rnd = interpolate.splrep(s,rnd,k=kx)
    tck_mnd = interpolate.splrep(s,mnd,k=kx)
    tck_gnd = interpolate.splrep(s,gnd,k=kx)

    # Return Interpolated Functions
    return tck_lnd,tck_mnd,tck_rnd,tck_gnd

