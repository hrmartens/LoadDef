# *********************************************************************
# FUNCTION TO CONVERT LAND-SEA MASK TO A SPARSE ARRAY
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
from scipy import meshgrid
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def main(lat,lon,lsmask,xx):

    # Define Footprint
    fp = np.ones((xx,xx))
    fp_size = fp.shape[0]*fp.shape[1]

    # Initialize New Arrays
    fill_lsm2darr = lsmask.copy()
    lsm2darr = lsmask.copy()

    # Fill Array with the Number 2 (Flag for Coastal Regions)
    fill_lsm2darr = np.ones((lsm2darr.shape[0],lsm2darr.shape[1]))*2.

    # Determine the Number of 1s Around Each Point
    counts = convolve2d(lsm2darr, fp, mode='same', boundary='wrap')

    # Replace +/-90 (Poles) with Original Values (Don't Want to 'Wrap Around' from Pole to Pole)
    # Multiplying by Size of the Footprint Ensures that Land Remains Land (9) and Ocean Remains Ocean (0)
    counts[0,:] = lsm2darr[0,:]*fp_size
    counts[-1,:] = lsm2darr[-1,:]*fp_size

    # Find Cell Blocks that Are Exclusively Ocean or Land, or Coastal
    myocean = np.where(counts == 0);       myoceanx = myocean[0]; myoceany = myocean[1]
    myland  = np.where(counts == fp_size); mylandx  = myland[0];  mylandy  = myland[1]

    # Overwrite Land and Ocean Nodes
    fill_lsm2darr[myoceanx,myoceany] = 0
    fill_lsm2darr[mylandx, mylandy]  = 1

    # Plot
    #plt.imshow(fill_lsm2darr)
    #plt.gca().invert_yaxis()
    #plt.show()

    # Reformat Elevation Points into 1D Vectors
    grid_olon, grid_olat = meshgrid(lon,lat)
    olon = grid_olon.flatten()
    olat = grid_olat.flatten()
    lsmask  = lsmask.flatten()
    fill_lsm2darr = fill_lsm2darr.flatten()

    # Find Where the Fill Values are Flagged by the Number 2 (Coastal Node); Delete All Other Nodes
    not_coast = np.where(fill_lsm2darr != 2); not_coast = not_coast[0]
    olon = np.delete(olon,not_coast)
    olat = np.delete(olat,not_coast)
    lsmask = np.delete(lsmask,not_coast)

    # Return Variables
    return olat,olon,lsmask

