# *********************************************************************
# FUNCTION TO CONVOLVE LOAD GREEN'S FUNCTIONS WITH A MASS-LOAD MODEL
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
import scipy as sc
from scipy import interpolate
from CONVGF.utility import read_AmpPha
from CONVGF.CN import convolve_global_grid
from CONVGF.CN import interpolate_load
from CONVGF.CN import coef2amppha
from CONVGF.CN import mass_conservation
import sys
import os
from math import pi

def main(loadfile,ilat,ilon,iarea,load_density,ur,ue,un,lsmk,mask,mydt,regular,mass_cons,lf_format):

    # Read the File
    llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_AmpPha.main(loadfile,lf_format,regular_grid=regular)
 
    # Find Where Amplitude is NaN (if anywhere) and Set to Zero
    nanidx = np.isnan(amp); amp[nanidx] = 0.; pha[nanidx] = 0.
 
    # Convert Amp/Pha Arrays to Real/Imag
    real = np.multiply(amp,np.cos(np.multiply(pha,pi/180.)))
    imag = np.multiply(amp,np.sin(np.multiply(pha,pi/180.)))

    # Interpolate Load at Each Grid Point onto the Integration Mesh
    ic1,ic2   = interpolate_load.main(ilat,ilon,llat,llon,real,imag,regular)

    # Multiply the Load Heights by the Load Density
    ic1 = np.multiply(ic1,load_density)
    ic2 = np.multiply(ic2,load_density)

    # Enforce Mass Conservation
    if (mass_cons == True):
        if (mask == 1): # For Oceans
            print(':: Warning: Enforcing Mass Conservation Over Oceans.')
            ic1_mc,ic2_mc = mass_conservation.main(ic1[lsmk==0],ic2[lsmk==0],iarea[lsmk==0])
            ic1[lsmk==0] = ic1_mc
            ic2[lsmk==0] = ic2_mc
        else: # For Land and Whole-Globe Models (like atmosphere and continental water)
            print(':: Warning: Enforcing Mass Conservation Over Entire Globe.')
            ic1,ic2 = mass_conservation.main(ic1,ic2,iarea)

    # Apply Land-Sea Mask Based on LS Mask Database (LAND=1;OCEAN=0) 
    # If mask = 2, Set Oceans to Zero (retain land)
    # If mask = 1, Set Land to Zero (retain ocean)
    # Else, Do Nothing (retain full model)
    if (mask == 2):
        ic1[lsmk == 0] = 0.
        ic2[lsmk == 0] = 0. 
    elif (mask == 1):
        ic1[lsmk == 1] = 0.
        ic2[lsmk == 1] = 0.

    # Perform the Convolution at Each Grid Point
    c1e,c2e,c1n,c2n,c1v,c2v = convolve_global_grid.main(ic1,ic2,ur,ue,un)

    # Sum Over All Grid Cells
    ec1 = np.sum(c1e)
    ec2 = np.sum(c2e)
    nc1 = np.sum(c1n)
    nc2 = np.sum(c2n)
    vc1 = np.sum(c1v)
    vc2 = np.sum(c2v)

    # Convert Coefficients to Amplitude and Phase
    eamp,epha,namp,npha,vamp,vpha = coef2amppha.main(ec1,ec2,nc1,nc2,vc1,vc2)

    # Return Variables
    return eamp,epha,namp,npha,vamp,vpha


