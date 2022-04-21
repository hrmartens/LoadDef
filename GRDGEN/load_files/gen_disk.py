#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO GENERATE A CUSTOM LOAD GRID
# :: GRIDS GENERATED MAY BE USED BY LOADDEF (run_cn.py) OR IN GMT
#
# Copyright (c) 2014-2017, CALIFORNIA INSTITUTE OF TECHNOLOGY
# HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
#
# DISCLAIMER: 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGE.
#
# *********************************************************************

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd() + "/../../")

# IMPORT PYTHON MODULES
import numpy as np
import scipy as sc
import netCDF4
from scipy import interpolate
from CONVGF.utility import read_AmpPha

# --------------- SPECIFY USER INPUTS --------------------- #

# Define the disk radius in km
disk_radius = 10 # in km

# compute disk latitude in degrees (90 degrees lat at north pole, minus the disk radius divided by the km in 1 angular degree (pi * diameter / 360))
disk_lat = 90 - (disk_radius / ((2*6371*np.pi)/360)) # latitude of the edge of the disk; north of here will have load, and south of here will have no load
disk_lat = np.around(disk_lat,decimals=4) # round the disk latitude to 4 decimal places

# Grid Spacing
gspace = 0.0002
#gspace = 0.1 

# Specify Bounding Box for Load Model (e.g. boundingbox.klokantech.com)
#  :: In general, the longitude range should be [0,360]
#  :: In the special case that the bounding box crosses the prime meridian,
#     the range should be [-180,0] for wlon and [0,180] for elon
wlon=0. # range [0,360] | If Bounding Box Crosses Prime Meridian, range = [-180,0]
elon=360. # range [0,360] | If Bounding Box Crosses Prime Meridian, range = [0,180]
slat=(disk_lat - (gspace*4))  # range [-90,90] [Slightly south of the disk edge (make sure the grid resolution, set below, gets a row or two of high-res points here)]
nlat=(disk_lat + (gspace*4))  # range [-90,90] [Slightly north of the disk edge (make sure the grid resolution, set below, gets a row or two of high-res points here)]
slat = np.around(slat,decimals=4)
nlat = np.around(nlat,decimals=4)

# Apply Prime-Meridian Correction?
#  :: Set to "True" if the Bounding Box Stradles the Prime Meridian
#  :: Ranges Must be [-180,0] and [0,180]
pm_correct = False

# Specify Load Height (meters)
loadamp = 5
#loadamp=0.203718 # Parsons
 
# Specify Phase (deg)
loadpha=0.0

# Apply the Given Load Height and Phase to the Region Outside the Regular Bounding Box
set_outside = False

# Apply the Given Load Height and Phase to the Region Inside the Regular Bounding Box
set_inside = True

# Taper at edges? Apply taper within how many degrees of edge?
taper = False
edge = 1.
taper_type = 'linear' # scipy interp1d: linear, nearest, zero, slinear, quadratic, cubic, etc.

# Optional: Starting Grid File (If no initial starting grid, set "initial_grid = None")
initial_grid = None
regular_grid = True

# Output Filename
outfile = ("disk_" + str(loadamp) + "m_" + str(disk_radius) + "km-NoTaper")
#outfile = ("disk_" + str(int(loadamp)) + "m_" + str(disk_radius) + "km-NoTaper")
  
# Write Load Information to a netCDF-formatted File? (Default for convolution)
write_nc = True

# Write Load Information to a Text File? (Alternative for convolution)
write_txt = False

# Write Load Information to a GMT-formatted File? (Lon, Lat, Amplitude)
write_gmt = False

# ------------------ END USER INPUTS ----------------------- #

# -------------------- BEGIN CODE -------------------------- #

# Check for output of a file
if (write_nc == False) and (write_txt == False) and (write_gmt == False):
    print(":: Error: No output file(s) selected. Options: netCDF, GMT, and/or plain-text.")
    sys.exit()

# Create Folders
if not (os.path.isdir("../../output/Grid_Files/")):
    os.makedirs("../../output/Grid_Files/")
if not (os.path.isdir("../../output/Grid_Files/GMT/")):
    os.makedirs("../../output/Grid_Files/GMT/")
if not (os.path.isdir("../../output/Grid_Files/GMT/Custom/")):
    os.makedirs("../../output/Grid_Files/GMT/Custom/")
if not (os.path.isdir("../../output/Grid_Files/nc/")):
    os.makedirs("../../output/Grid_Files/nc/")
if not (os.path.isdir("../../output/Grid_Files/nc/Custom/")):
    os.makedirs("../../output/Grid_Files/nc/Custom/")
if not (os.path.isdir("../../output/Grid_Files/text/")):
    os.makedirs("../../output/Grid_Files/text/")
if not (os.path.isdir("../../output/Grid_Files/text/Custom/")):
    os.makedirs("../../output/Grid_Files/text/Custom/")

# Read in Starting Grid, or Generate a New Grid
if initial_grid is not None:
    # Read in Starting Grid
    llat,llon,amp,pha,lat1dseq,lon1dseq,amp2darr,pha2darr = read_AmpPha.main(initial_grid,regular_grid=regular_grid)
# Generate New Grid
else:
    lats = np.arange(slat,nlat,gspace) 
    lons = np.arange(wlon,elon,gspace)
    xv,yv = np.meshgrid(lons,lats)
    llon = np.ravel(xv)
    llat = np.ravel(yv)
    llat = np.around(llat,decimals=8)
    llon = np.around(llon,decimals=8)
    amp = np.zeros((len(llon),))
    pha = np.zeros((len(llon),))

# If Necessary, Apply Prime-Meridian Correction (Shift to Range [-180,180])
if (pm_correct == True):
    print(':: Applying the prime-meridian correction.')
    if wlon > 0.:
        sys.exit('Error: When applying the prime-meridian correction, the longitudes of the bounding box must range from [-180,180].')
    if elon < 0.:
        sys.exit('Error: When applying the prime-meridian correction, the longitudes of the bounding box must range from [-180,180].')
    pm_correction = np.where(llon>=180.); pm_correction = pm_correction[0]
    llon[pm_correction] -= 360.

# Find Indices Outside Disk
bboxoutside = np.where((llat <= disk_lat)); bboxoutside = bboxoutside[0]

# Find Indices Within Disk
bboxinside = np.where((llat >= disk_lat)); bboxinside = bboxinside[0]

# If Necessary, Shift Longitude Values back to Original Range ([0,360])
if (pm_correct == True):
    llon[pm_correction] += 360.

# Set Amplitude and Phase to Constant Values Outside Bounding Box
if (set_outside == True):
    amp[bboxoutside] = loadamp
    pha[bboxoutside] = loadpha
    # Apply taper
    if taper:
        print(':: Tapering around edges.')
        # South Pole
        func = interpolate.interp1d([slat-edge,slat],[loadamp,0.],kind=taper_type)
        taper_idx = np.where((llat >= (slat-edge)) & (llat <= slat)); taper_idx = taper_idx[0]
        taper_amp = func(llat[taper_idx])
        print(taper_amp)
        amp[taper_idx] = taper_amp.copy()
        # North Pole
        func = interpolate.interp1d([nlat,nlat+edge],[0.,loadamp],kind=taper_type)
        taper_idx = np.where((llat <= (nlat+edge)) & (llat >= nlat)); taper_idx = taper_idx[0]
        taper_amp = func(llat[taper_idx])
        amp[taper_idx] = taper_amp.copy()

# Set Amplitude and Phase to Constant Values Inside Bounding Box
if (set_inside == True):
    amp[bboxinside] = loadamp
    pha[bboxinside] = loadpha

# Output OTL Grid to File for Plotting in GMT
if (write_gmt == True):
    print(":: Writing GMT-convenient text file.")
    custom_out = outfile + ".txt"
    custom_file = ("../../output/Grid_Files/GMT/Custom/height-anomaly_" + custom_out)
    # Prepare Data
    all_custom_data = np.column_stack((llon,llat,amp))
    # Write Data to File
    np.savetxt(custom_file, all_custom_data, fmt='%f %f %f')

# Output Grid to File for Use with LoadDef
if (write_nc == True):
    print(":: Writing netCDF-formatted file.")
    custom_out = (outfile + ".nc")
    custom_file = ("../../output/Grid_Files/nc/Custom/convgf_" + custom_out)
    # Open new NetCDF file in "write" mode
    dataset = netCDF4.Dataset(custom_file,'w',format='NETCDF4_CLASSIC')
    # Define dimensions for variables
    num_pts = len(llat)
    latitude = dataset.createDimension('latitude',num_pts)
    longitude = dataset.createDimension('longitude',num_pts)
    amplitude = dataset.createDimension('amplitude',num_pts)
    phase = dataset.createDimension('phase',num_pts)
    # Create variables
    latitudes = dataset.createVariable('latitude',float,('latitude',))
    longitudes = dataset.createVariable('longitude',float,('longitude',))
    amplitudes = dataset.createVariable('amplitude',float,('amplitude',))
    phases = dataset.createVariable('phase',float,('phase',))
    # Add units
    latitudes.units = 'degree_north'
    longitudes.units = 'degree_east'
    amplitudes.units = 'm'
    phases.units = 'degree'
    # Assign data
    latitudes[:] = llat
    longitudes[:] = llon
    amplitudes[:] = amp
    phases[:] = pha
    # Write Data to File
    dataset.close()
if (write_txt == True):
    print(":: Writing plain-text file.")
    custom_out = (outfile + ".txt")
    custom_file = ("../../output/Grid_Files/text/Custom/convgf_" + custom_out)
    # Prepare Data
    all_custom_data = np.column_stack((llat,llon,amp,pha))
    # Write Data to File
    np.savetxt(custom_file, all_custom_data, fmt='%f %f %f %f')

# --------------------- END CODE --------------------------- #


