# *********************************************************************
# PROGRAM TO GENERATE A SPARSE LAND-SEA MASK FROM ETOPO1
# https://www.ngdc.noaa.gov/mgg/global/global.html
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

# Import Python Modules
from __future__ import print_function
import numpy as np
from GRDGEN.utility import read_etopo1_sparse
from GRDGEN.utility import read_add_ascii
import matplotlib.pyplot as plt
import sys
import os

def main(landsea_dir,etopo1_file,outdir,antarctic_coastline=None,show_figures=False):

    # Create Folders
    if not (os.path.isdir(outdir)):
        os.makedirs(outdir)

    # LAND = 1 ; OCEAN = 0

    # Generate Mask from ETOPO1
    print(':: Generating the Sparse Land-Sea Mask. This May Take a Few Minutes. Please Wait...')
    lslat,lslon,lsmask,lslat1dseq,lslon1dseq,lsmask2darr = read_etopo1_sparse.main(landsea_dir + etopo1_file,show_figures=show_figures)
    print(':: Generation of the Land-Sea Mask Has Completed Successfully.')
    print(':: Size of LSMask: %s' %(str(lsmask.shape)))

    # Plot
    if (show_figures == True):
        ocean = np.where(lsmask == 0.); ocean = ocean[0]
        land  = np.where(lsmask == 1.); land = land[0]
        plt.plot(lslon[ocean],lslat[ocean],'.',color='c',ms=1)
        plt.plot(lslon[land],lslat[land],'.',color='m',ms=1)
        plt.show()

    # Ensure that Land-Sea Mask Longitudes are in Range 0-360
    neglon_idx = np.where(lslon<0.)
    lslon[neglon_idx] = lslon[neglon_idx] + 360.

    # Filename
    filename = etopo1_file.split('.'); outfile = (outdir + filename[0] + '.txt')

    # Optionally, Apply the Antarctic Digital Database Coastlines Around Antarctica
    if antarctic_coastline is not None:
        if (os.path.isfile(antarctic_coastline)):
            if (os.stat(antarctic_coastline).st_size != 0):
                print(':: Refining Antarctic Coastline. Please Wait...')
                # Locate and Delete Land-Sea Mask Values South of -60S
                ant_idx = np.where(lslat <= -60.); ant_idx = ant_idx[0]
                rev_lslat = np.delete(lslat,ant_idx); rev_lslon = np.delete(lslon,ant_idx); rev_lsmask = np.delete(lsmask,ant_idx)
                # Replace with ADD Coastlines
                add_lat,add_lon,add_lsmask = read_add_ascii.main(antarctic_coastline)
                lslat = np.concatenate((rev_lslat,add_lat))
                lslon = np.concatenate((rev_lslon,add_lon))
                lsmask = np.concatenate((rev_lsmask,add_lsmask))

                # Plot
                if (show_figures == True):
                    ocean = np.where(lsmask == 0.); ocean = ocean[0]
                    land  = np.where(lsmask == 1.); land = land[0]
                    plt.plot(lslon[ocean],lslat[ocean],'.',color='c',ms=1)
                    plt.plot(lslon[land],lslat[land],'.',color='m',ms=1)
                    plt.show()

                # Filename
                filename = etopo1_file.split('.'); outfile = (outdir + filename[0] + '_wADD.txt')

    # Save the LS Mask to a File 
    data = np.column_stack((lslat,lslon,lsmask))
    #f_handle = open(outfile, 'w')
    #np.savetxt(f_handle, data, fmt='%f %f %d')
    #f_handle.close()
    np.savetxt(outfile, data, fmt='%f %f %d')


