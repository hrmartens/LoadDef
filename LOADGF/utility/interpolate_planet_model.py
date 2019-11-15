# *********************************************************************
# FUNCTION TO INTERPOLATE THE INPUT PLANETARY MODEL
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
from scipy import interpolate 
import sys

def main(r,mu,K,lmda,rho,interval,kx=1,startinterp=None,stopinterp=None):

    # Starting and Stopping Values
    if startinterp is None:
        startinterp = min(r)
    if stopinterp is None:
        stopinterp = max(r)

    # Now Interpolate the Model
    cr = np.arange(startinterp,stopinterp,interval)
    tck_mu = interpolate.splrep(r,mu,k=kx)
    cmu = interpolate.splev(cr,tck_mu,der=0)
    tck_K = interpolate.splrep(r,K,k=kx)
    cK = interpolate.splev(cr,tck_K,der=0)
    tck_lmda = interpolate.splrep(r,lmda,k=kx)
    clmda = interpolate.splev(cr,tck_lmda,der=0)
    tck_rho = interpolate.splrep(r,rho,k=kx)
    crho = interpolate.splev(cr,tck_rho,der=0)
  
    # Convert Lists Back to Arrays
    cmu = np.asarray(cmu)
    cK = np.asarray(cK)
    crho = np.asarray(crho)
    clmda = np.asarray(clmda)
    cr = np.asarray(cr)
    
    # Return Interpolated Values
    return cr,cmu,cK,clmda,crho

