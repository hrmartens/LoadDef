#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO PLOT LOAD MODELS
# INPUTS: MODEL MAY BE IN THE FORMAT [Lat, Lon, Amp, Pha] OR [Lat, Lon, Value]
# "VALUE" MAY OPTIONALLY BE BINARY
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

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd() + "/../../")
 
# Import Python Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata
from math import ceil
from CONVGF.utility import read_AmpPha
from CONVGF.utility import read_lsmask
 
# Load or Land-Sea File
infile = ("../../output/Grid_Files/nc/OTL/convgf_TPXO8-Atlas-M2.nc") # Example 1
#infile = ("../../input/Land_Sea/ETOPO1_Ice_g_gmt4_wADD.txt") # Example 2

# Format for the Load or Land-Sea File. Acceptable formats include: netCDF ("nc") and plain-text ("txt").
file_format = "nc"

# Is the grid regular?
regular_grid = True # Example 1
#regular_grid = False # Example 2 
 
# Does the Grid Include both Amplitude and Phase Values? 
amp_and_pha = True # Example 1
#amp_and_pha = False # Example 2 

# Plot real or imaginary instead of amplitude? (only one, or neither, may be true)
plot_real = False
plot_imag = False

# Binary Data? (e.g., Land-Sea Mask)
binary = False # Example 1
#binary = True # Example 2

# Output Figure Name
figname = ("load_model.pdf") # Example 1
#figname = ("landsea_mask.pdf") # Example 2
 
# Min and Max Values for the Colormap
cmin = 0.
cmax = 1.

# Interpolate the grid? (If False, Plot Points as they Are)
#  :: Only applies to non-regular grids
interp_grid = False

# Color Map
colormap = cm.BuPu

#### BEGIN CODE

# Create Folder
if not (os.path.isdir("./output/")):
    os.makedirs("./output/")
outdir = "./output/"

# Read Grid
if (amp_and_pha == True):
    llat,llon,amp,pha,llat1darr,llon1darr,amp2darr,pha2darr = read_AmpPha.main(infile,file_format,delim=None,regular_grid=regular_grid)
    real2darr = np.multiply(amp2darr,np.cos(np.multiply(pha2darr,(np.pi/180.))))
    imag2darr = np.multiply(amp2darr,np.sin(np.multiply(pha2darr,(np.pi/180.))))
    if plot_real:
        amp2darr = real2darr.copy()
    if plot_imag:
        amp2darr = imag2darr.copy()
else:    
    llat,llon,amp = read_lsmask.main(infile,delim=None) 
    if (regular_grid == True):
        # Save Arrays in Grid Format
        lon1dseq = np.unique(llon)
        lat1dseq = np.unique(llat)
        amp2darr = np.empty((len(lat1dseq),len(lon1dseq)))
        # Determine Indices of Unique 1d Arrays that Correspond to Lat/Lon Values in Original 1d Arrays
        myidxlat = np.searchsorted(lat1dseq,llat)
        myidxlon = np.searchsorted(lon1dseq,llon)
        amp2darr[myidxlat,myidxlon] = amp

# Mapping Function
# See: http://scipy-cookbook.readthedocs.io/items/Matplotlib_Gridding_irregularly_spaced_data.html
def show_map(lon,lat,amp,spatial_resolution):

    # Initialize Figure
    fig = plt.figure()

    # Interpolate Grid
    if (interp_grid == True):

        # Build the Regular Grid
        lat_min = min(lat)
        lat_max = max(lat)
        lon_min = min(lon)
        lon_max = max(lon)
        data_min = min(amp)
        data_max = max(amp)
        yinum = (lat_max - lat_min) / spatial_resolution
        xinum = (lon_max - lon_min) / spatial_resolution
        yi = np.linspace(lat_min, lat_max, ceil(yinum), endpoint=True)        # same as [lat_min:spatial_resolution:lat_max] in matlab
        xi = np.linspace(lon_min, lon_max, ceil(xinum), endpoint=True)        # same as [lon_min:spatial_resolution:lon_max] in matlab
        xi, yi = np.meshgrid(xi, yi)

        # Grid the Irregularly Spaced Data
        zi = griddata((lon, lat), amp, (xi, yi), method='linear')

        # Plot with imshow
        plt.imshow(zi,cmap=colormap,origin="lower",clim=(cmin, cmax))
        plt.colorbar(orientation='horizontal')
        plt.savefig((outdir+figname),orientation='landscape',format='pdf')
        plt.show()

    # Plot points as they are
    else:

        # Binary Data
        if (binary == True):
            ocean = np.where(amp == 0.); ocean = ocean[0]
            land  = np.where(amp == 1.); land = land[0]
            plt.plot(lon[ocean],lat[ocean],'.',color='c',ms=1)
            plt.plot(lon[land],lat[land],'.',color='m',ms=1)
            plt.savefig((outdir+figname),orientation='landscape',format='pdf')
            plt.show()
        else:
            # Plot data
            plt.scatter(lon,lat,c=amp,s=1,cmap=colormap)
            plt.colorbar(orientation='horizontal')
            plt.savefig((outdir+figname),orientation='landscape',format='pdf')
            plt.show()

# Plot map
print(':: Plotting map')
if (regular_grid == True):
    plt.imshow(amp2darr,cmap=colormap,origin="lower",clim=(cmin, cmax)) 
    plt.colorbar(orientation='horizontal')
    plt.savefig((outdir+figname),orientation='landscape',format='pdf')
    plt.show()
else:
    spacing = 0.0625
    show_map(llon,llat,amp,spacing)

