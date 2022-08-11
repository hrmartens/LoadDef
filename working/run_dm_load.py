#!/usr/bin/env python

# *********************************************************************
# MAIN PROGRAM TO COMPUTE A DESIGN MATRIX TO INVERT FOR SURFACE LOAD --
# BY CONVOLVING DISPLACEMENT LOAD GREENS FUNCTIONS WITH A UNIFORM LOAD IN 
# EACH USER-DEFINED GRID CELL 
#
# Copyright (c) 2014-2022: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
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

# IMPORT PRINT FUNCTION
from __future__ import print_function

# IMPORT MPI MODULE
from mpi4py import MPI

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
import sys
import os
sys.path.append(os.getcwd() + "/../")

# IMPORT PYTHON MODULES
import numpy as np
import scipy as sc
import datetime
import netCDF4 
from CONVGF.CN import load_convolution
from CONVGF.utility import read_station_file
from CONVGF.utility import read_lsmask

# --------------- SPECIFY USER INPUTS --------------------- #

# Reference Frame (used for filenames) [Blewitt 2003]
rfm = "cf"

# Greens Function File
#  :: May be load Green's function file output directly from run_gf.py (norm_flag = False)
#  :: May be from a published table, normalized according to Farrell (1972) conventions [theta, u_norm, v_norm]
pmod = "PREM"
grn_file = ("../output/Greens_Functions/" + rfm + "_" + pmod + ".txt")
norm_flag  = False

# Full Path to Grid File Containing Cells
#  :: Format: south lat [float], north lat [float], west lon [float], east lon [float], unique cell id [string]
gridname = ("cells_44.0_46.0_247.0_249.0_0.25")
#gridname = ("cells_31.0_49.5_234.0_256.0_0.25")
loadgrid = ("../output/Grid_Files/nc/cells/" + gridname + ".nc") 

# Load Density
#  Recommended: 1000 kg/m^3 as a standard for inversion
ldens = 1000.0
  
# Ocean/Land Mask 
#  :: 0 = do not mask ocean or land (retain full model); 1 = mask out land (retain ocean); 2 = mask out oceans (retain land)
#  :: Recommended: 1 for oceanic; 2 for atmospheric and continental water
lsmask_type = 0

# Full Path to Land-Sea Mask File (May be Irregular and Sparse)
#  :: Format: Lat, Lon, Mask [0=ocean; 1=land]
lsmask_file = ("../input/Land_Sea/ETOPO1_Ice_g_gmt4_wADD.txt")

# Station/Grid-Point Location File (Lat, Lon, StationName)
sta_file_name = ("NOTA_Select")
sta_file = ("../input/Station_Locations/" + sta_file_name + ".txt")

# Optional: Additional string to include in output filenames (e.g. "_2019")
outstr = (pmod + "_" + gridname + "_" + sta_file_name)

# ------------------ END USER INPUTS ----------------------- #

# -------------------- SETUP MPI --------------------------- #

# Get the Main MPI Communicator That Controls Communication Between Processors
comm = MPI.COMM_WORLD
# Get My "Rank", i.e. the Processor Number Assigned to Me
rank = comm.Get_rank()
# Get the Total Number of Other Processors Used
size = comm.Get_size()

# ---------------------------------------------------------- #

# -------------------- BEGIN CODE -------------------------- #

# LoadFile Format ("bbox" tells the software to read the text file line by line for individual bounding-box cells in the load grid)
loadfile_format = "bbox"

# Bounding boxes for grid cells are regular in lat/lon
regular = True

# Check for existence of load grid
if not os.path.isfile(loadgrid):
    sys.exit('Error: The load grid does not exist. You may need to create it.')

# Put loadgrid file into a list (for consistency with how traditional load files are treated)
load_files = []
load_files.append(loadgrid)

# Ensure that the Output Directories Exist
if (rank == 0):
    if not (os.path.isdir("../output/Convolution/")):
        os.makedirs("../output/Convolution/")
    if not (os.path.isdir("../output/DesignMatrixLoad/")):
        os.makedirs("../output/DesignMatrixLoad/")
    if not (os.path.isdir("../output/Figures/")):
        os.makedirs("../output/Figures/")

    # Read Station & Date Range File
    lat,lon,sta = read_station_file.main(sta_file)

    # Determine Number of Stations Read In
    if isinstance(lat,float) == True: # only 1 station
        numel = 1
    else:
        numel = len(lat)

    # Read in the Land-Sea Mask
    if (lsmask_type > 0):
        lslat,lslon,lsmask = read_lsmask.main(lsmask_file)
    else:
        # Doesn't really matter so long as there are some values filled in with something other than 1 or 2
        lat1d = np.arange(-90.,90.,2.)
        lon1d = np.arange(0.,360.,2.)
        olon,olat = np.meshgrid(lon1d,lat1d)
        lslat = olat.flatten()
        lslon = olon.flatten()
        lsmask = np.ones((len(lslat),)) * -1.

    # Ensure that Land-Sea Mask Longitudes are in Range 0-360
    neglon_idx = np.where(lslon<0.)
    lslon[neglon_idx] = lslon[neglon_idx] + 360.
 
    # Read in the loadgrid
    lcext = loadgrid[-2::]
    if (lcext == 'xt'):
        load_cells = np.loadtxt(loadgrid,usecols=(4,),unpack=True,dtype='U')
        lcslat,lcnlat,lcwlon,lcelon = np.loadtxt(loadgrid,usecols=(0,1,2,3),unpack=True)
    elif (lcext == 'nc'):
        f = netCDF4.Dataset(loadgrid)
        load_cells = f.variables['cell_ids'][:]
        lcslat = f.variables['slatitude'][:]
        lcnlat = f.variables['nlatitude'][:]
        lcwlon = f.variables['wlongitude'][:]
        lcelon = f.variables['elongitude'][:]
        f.close()    
    # Ensure that Bounding Box Longitudes are in Range 0-360
    for yy in range(0,len(lcwlon)):
        if (lcwlon[yy] < 0.):
            lcwlon[yy] += 360.
        if (lcelon[yy] < 0.):
            lcelon[yy] += 360.
    # Compute center of each load cell
    print(':: Warning: Computing center of load cells. Special consideration should be made for cells spanning the prime meridian, if applicable.')
    lclat = (lcslat + lcnlat)/2.
    lclon = (lcwlon + lcelon)/2.
# If I'm a Worker, I Know Nothing About the Data
else:
    load_cells = lclat = lclon = lslat = lslon = lsmask = sta = lat = lon = numel = None 

# All Processors Get Certain Arrays and Parameters; Broadcast Them
load_cells  = comm.bcast(load_cells, root=0)
lclat       = comm.bcast(lclat, root=0)
lclon       = comm.bcast(lclon, root=0)
lslat       = comm.bcast(lslat, root=0)
lslon       = comm.bcast(lslon, root=0)
lsmask      = comm.bcast(lsmask, root=0)
sta         = comm.bcast(sta, root=0)
lat         = comm.bcast(lat, root=0)
lon         = comm.bcast(lon, root=0)
numel       = comm.bcast(numel, root=0)

# Determine the Chunk Sizes for the Convolution
total_cells = len(load_cells)
nominal_load = total_cells // size # Floor Divide
# Final Chunk Might Be Different in Size Than the Nominal Load
if rank == size - 1:
    procN = total_cells - rank * nominal_load
else:
    procN = nominal_load

# Set up Design matrix (rows = stations[e,n,u]; columns = load cells)
if (rank == 0):
    desmat = np.zeros((numel*3, total_cells)) # Multiplication by 3 for 3 spatial dimensions (e,n,u)
    dmrows = np.empty((numel*3,),dtype='U10') # Assumes that station names are no more than 9 characters in length (with E, N, or U also appended)
    sclat = np.zeros((numel*3,))
    sclon = np.zeros((numel*3,))

# Loop Through Each Station
for jj in range(0,numel):

    # Remove Index If Only 1 Station
    if (numel == 1): # only 1 station read in
        my_sta = sta
        my_lat = lat
        my_lon = lon
    else:
        my_sta = sta[jj]
        my_lat = lat[jj]
        my_lon = lon[jj]

    # If Rank is Master, Output Station Name
    try: 
        my_sta = my_sta.decode()
    except: 
        if (rank == 0):
            print(':: No need to decode station.')
    if (rank == 0):
        print(' ')
        print(':: Starting on Station: ' + my_sta)

    # Output File Name
    cnv_out = (my_sta + "_" + rfm + "_" + outstr + ".txt")

    # Set Lat/Lon/Name for Current Station
    slat = my_lat
    slon = my_lon
    sname = my_sta

    # Adjust longitude, if necessary
    if (slon < 0.):
        slon += 360.

    # Perform the Convolution for Each Station
    #### NOTE: Mesh defaults are adjusted to ensure we get a good number of points within each grid cell to adequately represent the shape of each cell
    if (rank == 0):
        print(":: General Warning: Use caution when selecting mesh parameters. For small cells and distant stations, there is a possibility that no mesh points will lie within the distant cell. Please adapt settings to your application. [run_dm.py]")
        eamp,epha,namp,npha,vamp,vpha = load_convolution.main(grn_file,norm_flag,load_files,regular,lslat,lslon,lsmask,\
            slat,slon,sname,cnv_out,lsmask_type,loadfile_format,rank,procN,comm,load_density=ldens,azminc=0.5,delinc3=0.005,delinc4=0.02,delinc5=0.05)
    # For Worker Ranks, Run the Code But Don't Return Any Variables
    else:
        load_convolution.main(grn_file,norm_flag,load_files,regular,lslat,lslon,lsmask,\
            slat,slon,sname,cnv_out,lsmask_type,loadfile_format,rank,procN,comm,load_density=ldens,azminc=0.5,delinc3=0.005,delinc4=0.02,delinc5=0.05)

    # Make Sure All Jobs Have Finished Before Continuing
    comm.Barrier()
 
    if (rank == 0):

        # Convert Amp/Pha to Displacement
        edisp = np.multiply(eamp,np.cos(np.multiply(epha,(np.pi/180.))))
        ndisp = np.multiply(namp,np.cos(np.multiply(npha,(np.pi/180.))))
        udisp = np.multiply(vamp,np.cos(np.multiply(vpha,(np.pi/180.))))

        # Fill in Design Matrix
        idxe = (jj*3)+0
        idxn = (jj*3)+1
        idxu = (jj*3)+2
        desmat[idxe,:] = edisp
        desmat[idxn,:] = ndisp
        desmat[idxu,:] = udisp
        dmrows[idxe] = (sname + 'E')
        dmrows[idxn] = (sname + 'N')
        dmrows[idxu] = (sname + 'U')
        sclat[idxe] = slat
        sclat[idxn] = slat
        sclat[idxu] = slat
        sclon[idxe] = slon
        sclon[idxn] = slon
        sclon[idxu] = slon

# Write Design Matrix to File
if (rank == 0):
    print(":: Writing netCDF-formatted file.")
    f_out = ("designmatrix_" + rfm + "_" + outstr + ".nc")
    f_file = ("../output/DesignMatrixLoad/" + f_out)
    # Open new NetCDF file in "write" mode
    dataset = netCDF4.Dataset(f_file,'w',format='NETCDF4_CLASSIC')
    # Define dimensions for variables
    desmat_shape = desmat.shape
    num_rows = desmat_shape[0]
    num_cols = desmat_shape[1]
    nstacomp = dataset.createDimension('nstacomp',num_rows)
    nloadcell = dataset.createDimension('nloadcell',num_cols)
    nchars = dataset.createDimension('nchars',10)
    # Create variables
    sta_comp_id = dataset.createVariable('sta_comp_id','S1',('nstacomp','nchars'))
    load_cell_id = dataset.createVariable('load_cell_id','S1',('nloadcell','nchars'))
    design_matrix = dataset.createVariable('design_matrix',float,('nstacomp','nloadcell'))
    sta_comp_lat = dataset.createVariable('sta_comp_lat',float,('nstacomp',))
    sta_comp_lon = dataset.createVariable('sta_comp_lon',float,('nstacomp',))
    load_cell_lat = dataset.createVariable('load_cell_lat',float,('nloadcell',))
    load_cell_lon = dataset.createVariable('load_cell_lon',float,('nloadcell',))
    # Add units
    sta_comp_id.units = 'string'
    sta_comp_id.long_name = 'station_component_id'
    load_cell_id.units = 'string'
    load_cell_id.long_name = 'load_cell_id'
    design_matrix.units = 'mm'
    design_matrix.long_name = 'displacement_mm'
    sta_comp_lat.units = 'degrees_north'
    sta_comp_lat.long_name = 'station_latitude'
    sta_comp_lon.units = 'degrees_east'
    sta_comp_lon.long_name = 'station_longitude'
    load_cell_lat.units = 'degrees_north'
    load_cell_lat.long_name = 'loadcell_latitude'
    load_cell_lon.units = 'degrees_east'
    load_cell_lon.long_name = 'loadcell_longitude'
    # Assign data
    #  https://unidata.github.io/netcdf4-python/ (see "Dealing with Strings")
    #  sta_comp_id[:] = netCDF4.stringtochar(np.array(dmrows,dtype='S10'))
    #  load_cell_id[:] = netCDF4.stringtochar(np.array(load_cells,dtype='S10'))
    sta_comp_id._Encoding = 'ascii'
    sta_comp_id[:] = np.array(dmrows,dtype='S10')
    load_cell_id._Encoding = 'ascii'
    load_cell_id[:] = np.array(load_cells,dtype='S10')
    design_matrix[:,:] = desmat
    sta_comp_lat[:] = sclat
    sta_comp_lon[:] = sclon
    load_cell_lat[:] = lclat
    load_cell_lon[:] = lclon
    # Write Data to File
    dataset.close()

    # Read the netCDF file as a test
    f = netCDF4.Dataset(f_file)
    #print(f.variables)
    sta_comp_ids = f.variables['sta_comp_id'][:]
    load_cell_ids = f.variables['load_cell_id'][:]
    design_matrix = f.variables['design_matrix'][:]
    sta_comp_lat = f.variables['sta_comp_lat'][:]
    sta_comp_lon = f.variables['sta_comp_lon'][:]
    load_cell_lat = f.variables['load_cell_lat'][:]
    load_cell_lon = f.variables['load_cell_lon'][:]
    f.close()

# --------------------- END CODE --------------------------- #


