# *********************************************************************
# FUNCTION TO NON-DIMENSIONALIZE MATERIAL PARAMETERS
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

def main(r,lmda,rho,mu,L_sc,R_sc,T_sc):
    s = r/L_sc
    lnd = lmda/(R_sc*(L_sc**2)*(T_sc**(-2)))
    mnd = mu/(R_sc*(L_sc**2)*(T_sc**(-2)))
    rnd = rho/R_sc
    
    return s,lnd,mnd,rnd

