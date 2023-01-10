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
from CONVGF.utility import read_cMesh
from CONVGF.CN import convolve_global_grid
from CONVGF.CN import interpolate_load
from CONVGF.CN import interpolate_lsmask
from CONVGF.CN import coef2amppha
from CONVGF.CN import mass_conservation
import sys
import os
from math import pi
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main(loadfile,lf_format,ur,ue,un,load_density,ilat,ilon,iarea,lsmk,mask,regular,mass_cons,stname="Station"):

    # Using a common geographic mesh?
    if (lf_format == "common"): 

        # Read in the common mesh with the load and land-sea mask already applied
        ilat, ilon, ic1, ic2, iarea = read_cMesh.main(loadfile)
 
    else: # station-centered mesh

        # Check load file format 
        if (lf_format == "bbox"): # list of cells, rather than traditional load files

            # Select the Appropriate Cell ID
            cslat = loadfile[0]
            cnlat = loadfile[1]
            cwlon = loadfile[2]
            celon = loadfile[3]
            yes_idx = np.where((ilat >= cslat) & (ilat <= cnlat) & (ilon >= cwlon) & (ilon <= celon)); yes_idx = yes_idx[0]
            print(':: Number of convolution grid points within load cell: ', len(yes_idx))

            # Find ilat and ilon within cell
            ic1 = np.zeros(len(ilat),)
            ic2 = np.zeros(len(ilat),)
            ic1[yes_idx] = 1. # Amplitude of 1 (ic1 = 1 only inside cell), phase of zero (keep ic2 = 0 everywhere)

            # Optionally plot the load cell
            #### Set flag to "False" to turn off plotting of the load cell; "True" to turn it on
            plot_fig = False
            if plot_fig: 
                print(':: Plotting the load cell. [perform_convolution.py]')
                cslat_plot = cslat - 0.5
                cnlat_plot = cnlat + 0.5
                cwlon_plot = cwlon - 0.5
                celon_plot = celon + 0.5
                idx_plot = np.where((ilon >= cwlon_plot) & (ilon <= celon_plot) & (ilat >= cslat_plot) & (ilat <= cnlat_plot)); idx_plot = idx_plot[0]
                ilon_plot = ilon[idx_plot]
                ilat_plot = ilat[idx_plot]
                ic1_plot = ic1[idx_plot]
                plt.scatter(ilon_plot,ilat_plot,c=ic1_plot,s=1,cmap=cm.BuPu)
                plt.colorbar(orientation='horizontal')
                fig_name = ("../output/Figures/" + stname + "_" + str(cslat) + "_" + str(cnlat) + "_" + str(cwlon) + "_" + str(celon) + ".png")
                plt.savefig(fig_name,format="png")
                #plt.show()
                plt.close()
 
        else: # traditional load file
 
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
    # Note: Conversion to mm from meters also happens here!
    eamp,epha,namp,npha,vamp,vpha = coef2amppha.main(ec1,ec2,nc1,nc2,vc1,vc2)

    # Return Variables
    return eamp,epha,namp,npha,vamp,vpha


