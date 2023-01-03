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
from scipy import interpolate
from CONVGF.utility import read_station_file
from CONVGF.utility import read_lsmask
from CONVGF.utility import read_greens_fcn_file
from CONVGF.utility import read_greens_fcn_file_norm
from CONVGF.utility import normalize_greens_fcns
from CONVGF.CN import load_convolution
from CONVGF.CN import compute_specific_greens_fcns
from CONVGF.CN import generate_integration_mesh
from CONVGF.CN import intmesh2geogcoords
from CONVGF.CN import integrate_greens_fcns
from CONVGF.CN import compute_angularDist_azimuth
from CONVGF.CN import interpolate_lsmask
from CONVGF.CN import coef2amppha
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
cellfname = ("cells_28.0_50.0_233.0_258.0_0.25")
loadgrid  = ("../output/Grid_Files/nc/cells/" + cellfname + ".nc") 

# OPTIONAL: Provide a common geographic mesh? 
# If true, must provide the full path to a mesh file. If false, a station-centered grid will be created. 
common_mesh = False
# Full Path to Grid File Containing Surface Mesh (for sampling the load Green's functions)
#  :: Format: latitude midpoints [float,degrees N], longitude midpoints [float,degrees E], unit area of each patch [float,dimensionless (need to multiply by r^2)]
meshfname = ("commonMesh_28.0_50.0_233.0_258.0_0.01_0.01")
convmesh = ("../output/Grid_Files/nc/commonMesh/" + meshfname + ".nc")

# Planet Radius (in meters; used for Greens function normalization)
planet_radius = 6371000.

# Load Density
#  Recommended: 1000 kg/m^3 as a standard for water-mass inversion
ldens = 1000.0

# Ocean/Land Mask
#  :: 0 = do not mask ocean or land (retain full model); 1 = mask out land (retain ocean); 2 = mask out oceans (retain land)
#  :: Recommended: 1 for oceanic; 2 for atmospheric and continental water
#  :: When pre-generating a common mesh, a land-sea mask can already be applied. Thus, it is recommended to set lsmask_type = 0 here. 
lsmask_type = 0

# Full Path to Land-Sea Mask File (May be Irregular and Sparse)
#  :: Format: Lat, Lon, Mask [0=ocean; 1=land]
lsmask_file = ("../input/Land_Sea/ETOPO1_Ice_g_gmt4_wADD.txt")

# Station/Grid-Point Location File (Lat, Lon, StationName)
sta_file_name = ("NOTA_Select")
sta_file = ("../input/Station_Locations/" + sta_file_name + ".txt")

# Optional: Additional string to include in output filenames (e.g. "_2019")
outstr = (pmod + "_" + cellfname + "_" + meshfname + "_" + sta_file_name)

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
        lat1d = np.arange(-90.,90.,5.)
        lon1d = np.arange(0.,360.,5.)
        olon,olat = np.meshgrid(lon1d,lat1d)
        lslat = olat.flatten()
        lslon = olon.flatten()
        lsmask = np.ones((len(lslat),)) * -1.

    # Ensure that Land-Sea Mask Longitudes are in Range 0-360
    neglon_idx = np.where(lslon<0.)
    lslon[neglon_idx] = lslon[neglon_idx] + 360.
 
    # Read in the loadgrid
    #  Here, used for determining how to delegate MPI tasks - and - to determine and write out the center of each load cell
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

    # Read in the common convolution mesh, if applicable
    if (common_mesh == True): 
        print(':: Common Mesh True. Reading in ilat, ilon, iarea.')
        lcext = convmesh[-2::]
        if (lcext == 'xt'):
            ilat,ilon,unit_area = np.loadtxt(convmesh,usecols=(0,1,2),unpack=True)
            iarea = np.multiply(unit_area, planet_radius**2) # convert from unit area to true area of the spherical patch in m^2
        elif (lcext == 'nc'):
            f = netCDF4.Dataset(convmesh)
            ilat = f.variables['midpoint_lat'][:]
            ilon = f.variables['midpoint_lon'][:]
            unit_area = f.variables['unit_area_patch'][:]
            f.close()
            iarea = np.multiply(unit_area, planet_radius**2) # convert from unit area to true area of the spherical patch in m^2
    else:
        ilat = ilon = iarea = None

    # Determine the Land-Sea Mask: Interpolate onto Mesh
    if (common_mesh == True): 
        print(':: Common Mesh True. Applying Land-Sea Mask.')
        print(':: Number of Grid Points: %s | Size of LSMask: %s' %(str(len(ilat)), str(lsmask.shape)))
        lsmk = interpolate_lsmask.main(ilat,ilon,lslat,lslon,lsmask)
        print(':: Finished LSMask Interpolation.')
    else:
        lsmk = None

    # For a common mesh, can pre-determine the mesh points within each cell used for the inversion.
    # Also, apply the land-sea mask.
    if (common_mesh == True):
        ## Check load file format
        if not (loadfile_format == "bbox"): # list of cells, rather than traditional load files
            sys.exit(':: Error -- The loadfile format should be bbox for generating the design matrix for surface load distribution.')
        ## Prepare land-sea mask application
        if (lsmask_type == 2): 
            test_elements = np.where(lsmk == 0); test_elements = test_elements[0]
        elif (lsmask_type == 1): 
            test_elements = np.where(lsmk == 1); test_elements = test_elements[0]
        ## Select the Appropriate Cell ID
        count = 0 # initialize figure counter (don't want to get stuck in huge loop plotting figures...)
        colvals = np.empty((len(lclat),),dtype=object) # initialize array that will hold lists of column indices (corresponding to LGF mesh points in each grid cell)
        for qq in range(0,len(lclat)):
            clc = load_cells[qq]
            cslat = lcslat[qq]
            cnlat = lcnlat[qq]
            cwlon = lcwlon[qq]
            celon = lcelon[qq]
            ## Find ilat and ilon within cell
            yes_idx = np.where((ilat >= cslat) & (ilat <= cnlat) & (ilon >= cwlon) & (ilon <= celon)); yes_idx = yes_idx[0]
            print(':: Number of convolution grid points within load cell ', clc, ' of ', len(lclat), ': ', len(yes_idx))
            ## Apply land-sea mask
            if (lsmask_type == 2): 
                mask = np.isin(yes_idx, test_elements, assume_unique=True)
                idx_to_delete = np.nonzero(mask)
                yes_idx = np.delete(yes_idx,idx_to_delete)
            elif (lsmask_type == 1): 
                mask = np.isin(yes_idx, test_elements, assume_unique=True)
                idx_to_delete = np.nonzero(mask)
                yes_idx = np.delete(yes_idx,idx_to_delete)
            ## Write indices within cell to the main array
            colvals[qq] = yes_idx.tolist()
            #### OPTIONAL: Plot the load cell
            #### To suppress plotting, set max_count to 0
            #max_count = 0
            #if (len(yes_idx)>0):
            #    if (count < max_count):
            #        plot_fig = True
            #        count += 1
            #        print(':: Figure count: ', count)
            #    else:
            #        plot_fig = False
            #else:
            #    plot_fig = False
            #if plot_fig:
            #    print(':: Plotting the load cell. [run_dm_load.py]')
            #    cslat_plot = cslat - 0.5
            #    cnlat_plot = cnlat + 0.5
            #    cwlon_plot = cwlon - 0.5
            #    celon_plot = celon + 0.5
            #    idx_plot = np.where((ilon >= cwlon_plot) & (ilon <= celon_plot) & (ilat >= cslat_plot) & (ilat <= cnlat_plot)); idx_plot = idx_plot[0]
            #    ilon_plot = ilon[idx_plot]
            #    ilat_plot = ilat[idx_plot]
            #    ic1_plot = cic1[idx_plot]
            #    plt.scatter(ilon_plot,ilat_plot,c=ic1_plot,s=1,cmap=cm.BuPu)
            #    plt.colorbar(orientation='horizontal')
            #    fig_name = ("../output/Figures/" + str(cslat) + "_" + str(cnlat) + "_" + str(cwlon) + "_" + str(celon) + ".png")
            #    plt.savefig(fig_name,format="png")
            #    #plt.show()
            #    plt.close()
    else:
        colvals = None

# If I'm a Worker, I Know Nothing About the Data
else:
    load_cells = lclat = lclon = lslat = lslon = lsmask = sta = lat = lon = numel = None 
    ilat = ilon = iarea = lsmk = colvals = None

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
ilat        = comm.bcast(ilat, root=0)
ilon        = comm.bcast(ilon, root=0)
iarea       = comm.bcast(iarea, root=0)
lsmk        = comm.bcast(lsmk, root=0)

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

    # If not using a common mesh, then set up a station-centered grid
    if (common_mesh == False): 
        # Generate Integration Mesh
        print(':: Generating the Station-Centered Integration Mesh. Please Wait...')
        gldel,glazm,ldel,lazm,unit_area = generate_integration_mesh.main(azinc=0.5,delinc3=0.005,delinc4=0.02,delinc5=0.05)

    # Read in the Green's Functions
    if norm_flag == True:
        theta,u,v,unormFarrell,vnormFarrell = read_greens_fcn_file_norm.main(grn_file,rad)
    else:
        theta,u,v,unormFarrell,vnormFarrell = read_greens_fcn_file.main(grn_file)

    # Integrate the Green's Functions
    if (common_mesh == True):
        if (rank == 0):
            print(':: Common Mesh True. Computing specific Greens functions for common mesh.')
            # Normalize Green's According to Farrell Convention
            unorm = np.multiply(u,theta)*1E12*planet_radius
            vnorm = np.multiply(v,theta)*1E12*planet_radius
            # Interpolate Green's Functions
            tck_gfu = interpolate.splrep(theta,unorm,k=3)
            tck_gfv = interpolate.splrep(theta,vnorm,k=3)
            # Find Great-Circle Distances between Station and Grid Points in the Common Mesh
            delta,haz = compute_angularDist_azimuth.main(slat,slon,ilat,ilon)
            # Compute Integrated Greens Functions
            gfu = interpolate.splev(delta,tck_gfu,der=0)
            gfv = interpolate.splev(delta,tck_gfv,der=0)
            uint = iarea * gfu
            vint = iarea * gfv
            # Compute Greens Functions Specific to Receiver and Grid (Geographic Coordinates)
            ur,ue,un = compute_specific_greens_fcns.main(haz,uint,vint)
    else:  
        print(':: Common Mesh False. Computing specific Greens functions for station-centered mesh.')
        # Normalize Greens Functions (Agnew Convention)
        unorm,vnorm = normalize_greens_fcns.main(theta,u,v,planet_radius)
        # Interpolate Green's Functions
        tck_gfu = interpolate.splrep(theta,unorm,k=3)
        tck_gfv = interpolate.splrep(theta,vnorm,k=3)
        # Integrate Green's Functions for Station-Centered Grid
        uint,vint = integrate_greens_fcns.main(gldel,glazm,ldel,lazm,tck_gfu,tck_gfv)
        # Compute Geographic Coordinates of Integration Mesh Cell Midpoints
        ilat,ilon,iarea = intmesh2geogcoords.main(slat,slon,ldel,lazm,unit_area)
        # Compute Angular Distance and Azimuth at Receiver Due to Load
        delta,haz = compute_angularDist_azimuth.main(slat,slon,ilat,ilon)
        # Compute Greens Functions Specific to Receiver and Grid (Geographic Coordinates)
        ur,ue,un = compute_specific_greens_fcns.main(haz,uint,vint)
        # Determine the Land-Sea Mask: Interpolate onto Mesh
        print(':: Common Mesh False. Applying Land-Sea Mask.')
        print(':: Number of Grid Points: %s | Size of LSMask: %s' %(str(len(ilat)), str(lsmask.shape)))
        lsmk = interpolate_lsmask.main(ilat,ilon,lslat,lslon,lsmask)
        print(':: Finished LSMask Interpolation.')

    # Perform the Convolution for Each Station
    if (common_mesh == True):
        if (rank == 0): 
            print(':: Common Mesh True. Summing up integrated and specific LGFs within each cell directly.')
            numvals = len(colvals)
            eamp = np.zeros((numvals,)) # initialize an array of zeros for amplitude
            namp = np.zeros((numvals,)) # initialize an array of zeros for amplitude
            vamp = np.zeros((numvals,)) # initialize an array of zeros for amplitude
            epha = np.zeros((numvals,)) # phase is zero for non-periodic hydro loads
            npha = np.zeros((numvals,)) # phase is zero for non-periodic hydro loads
            vpha = np.zeros((numvals,)) # phase is zero for non-periodic hydro loads
            ec2 = 0 # imaginary component is zero for non-periodic load
            nc2 = 0 # imaginary component is zero for non-periodic load
            vc2 = 0 # imaginary component is zero for non-periodic load
            for dd in range(0,numvals): # loop through load cells
                myvals = colvals[dd] # Find indices of Greens-function mesh points within the current load cell
                if (len(myvals) == 0): # Nothing in this cell; skip it
                    continue
                else: # Sum up the contributions from each surface patch within the current load cell
                    ec1 = np.sum(ue[myvals])*ldens # Sum up all the relevant integrated and specific Greens functions (east); and then multiply by load density
                    nc1 = np.sum(un[myvals])*ldens # Sum up all the relevant integrated and specific Greens functions (north); and then multiply by load density
                    vc1 = np.sum(ur[myvals])*ldens # Sum up all the relevant integrated and specific Greens functions (up); and then multiply by load density
                    # Convert Coefficients to Amplitude and Phase 
                    # Note: Conversion from meters to mm also happens here!
                    ceamp,cepha,cnamp,cnpha,cvamp,cvpha = coef2amppha.main(ec1,ec2,nc1,nc2,vc1,vc2)
                    # Assign values to appropriate amplitude arrays
                    eamp[dd] = ceamp
                    namp[dd] = cnamp 
                    vamp[dd] = cvamp
    else:
        if (rank == 0): 
            print(':: Common Mesh False. Performing the standard convolution.')
        # For a station-centered grid, we cannot pre-determine the grid points within each cell, so we will send information to another function. 
        #### NOTE: Mesh defaults are adjusted to ensure we get a good number of points within each grid cell to adequately represent the shape of each cell.
        if (rank == 0):
            eamp,epha,namp,npha,vamp,vpha = load_convolution.main(ilat,ilon,iarea,ur,ue,un,regular,load_files,loadfile_format,lsmk,slat,slon,sname,cnv_out,\
                rank,procN,comm,load_density=ldens)
        # For Worker Ranks, Run the Code But Don't Return Any Variables
        else: 
            load_convolution.main(ilat,ilon,iarea,ur,ue,un,regular,load_files,loadfile_format,lsmk,slat,slon,sname,cnv_out,\
                rank,procN,comm,load_density=ldens)

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


