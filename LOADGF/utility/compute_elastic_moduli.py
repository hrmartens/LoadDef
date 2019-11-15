# *********************************************************************
# FUNCTION TO CONVERT SEISMIC VELOCITIES TO ELASTIC MODULI
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

def main(vp,vs,rho):
    mu = np.multiply(np.square(vs),rho)
    K  = np.subtract(np.multiply(np.square(vp),rho),mu*(4./3.))
    la = np.subtract(K, mu*(2./3.))
    return mu,K,la

