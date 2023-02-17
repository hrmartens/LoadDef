#!/usr/bin/env python

# *********************************************************************
# MAIN PROGRAM TO PREDICT SURFACE DISPLACEMENTS CAUSED BY SURFACE MASS LOADING 
# BY CONVOLVING DISPLACEMENT LOAD GREENS FUNCTIONS WITH A MODEL FOR A SURFACE MASS LOAD 
#
# Copyright (c) 2014-2023: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
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
from math import pi
from CONVGF.utility import read_station_file
from CONVGF.utility import read_lsmask
from CONVGF.utility import read_greens_fcn_file
from CONVGF.utility import read_greens_fcn_file_norm
from CONVGF.utility import normalize_greens_fcns
from CONVGF.utility import read_AmpPha
from CONVGF.CN import load_convolution
from CONVGF.CN import interpolate_load
from CONVGF.CN import compute_specific_greens_fcns
from CONVGF.CN import generate_integration_mesh
from CONVGF.CN import intmesh2geogcoords
from CONVGF.CN import integrate_greens_fcns
from CONVGF.CN import compute_angularDist_azimuth
from CONVGF.CN import interpolate_lsmask
from CONVGF.CN import coef2amppha
from CONVGF.CN import mass_conservation

# --------------- SPECIFY USER INPUTS --------------------- #

# Reference Frame (used for filenames) [Blewitt 2003]
rfm = "ce"

# Greens Function File
#  :: May be load Green's function file output directly from run_gf.py (norm_flag = False)
#  :: May be from a published table, normalized according to Farrell (1972) conventions [theta, u_norm, v_norm]
pmod = ("PREM")
grn_file = ("../output/Greens_Functions/" + rfm + "_" + pmod + ".txt")
norm_flag  = False
 
# Full Path to Load Directory and Prefix of Filename
loadfile_directory = ("../output/Grid_Files/nc/Custom/")

# Prefix for the Load Files (Load Directory will be Searched for all Files Starting with this Prefix)
#  :: Note: For Load Files Organized by Date, the End of Filename Name Must be in the Format yyyymmddhhmnsc.txt
#  :: Note: If not organized by date, files may be organized by tidal harmonic, for example (i.e. a unique filename ending)
#  :: Note: Output names (within output files) will be determined by extension following last underscore character (e.g., date/harmonic/model)
loadfile_prefix = ("convgf_disk_1m")

# LoadFile Format: ["nc", "txt"]
loadfile_format = "nc"
 
# Are the Load Files Organized by Datetime?
#  :: If False, all Files that match the loadfile directory and prefix will be analyzed.
time_series = False 

# Date Range for Computation (Year,Month,Day,Hour,Minute,Second)
#  :: Note: Only used if 'time_series' is True
frst_date = [2015,1,1,0,0,0]
last_date = [2016,3,1,0,0,0]

# Are the load values on regular grids (speeds up interpolation); If unsure, leave as false.
regular = True

# Load Density
#  Recommended: 1025-1035 for oceanic loads (e.g., FES2014, ECCO2); 1 for atmospheric loads (e.g. ECMWF)
ldens = 1000.0

# NEW OPTION: Provide a common geographic mesh?
# If True, must provide the full path to a mesh file (see: GRDGEN/common_mesh). 
# If False, a station-centered grid will be created within the functions called here. 
common_mesh = True
# Full Path to Grid File Containing Surface Mesh (for sampling the load Green's functions)
#  :: Format: latitude midpoints [float,degrees N], longitude midpoints [float,degrees E], unit area of each patch [float,dimensionless (need to multiply by r^2)]
meshfname = ("commonMesh_global_1.0_1.0_85.0_90.0_0.0_360.0_0.1_0.1_89.5_90.0_0.0_360.0_0.005_0.005")
convmesh = ("../output/Grid_Files/nc/commonMesh/" + meshfname + ".nc")

# Planet Radius (in meters; used for Greens function normalization)
planet_radius = 6371000.
  
# Ocean/Land Mask 
#  :: 0 = do not mask ocean or land (retain full model); 1 = mask out land (retain ocean); 2 = mask out oceans (retain land)
#  :: Recommended: 1 for oceanic; 2 for atmospheric
lsmask_type = 0

# Full Path to Land-Sea Mask File (May be Irregular and Sparse)
#  :: Format: Lat, Lon, Mask [0=ocean; 1=land]
lsmask_file = ("../input/Land_Sea/ETOPO1_Ice_g_gmt4_wADD.txt")
 
# Enforce mass conservation by removing a spatial mean from the load grid?
mass_cons = False

# Station/Grid-Point Location File (Lat, Lon, StationName)
sta_file = ("../input/Station_Locations/Lat_Profile_Select.txt")

# -- Mesh Paramters -- High Resolution
#del1 = 0.001    # increment in angular resolution (degrees) for innermost zone
#del2 = 0.005    # increment in angular resolution for second zone
#del3 = 0.01     # increment in angular resolution for third zone
#del4 = 0.1      # increment in angular resolution for fourth zone
#del5 = 0.5      # increment in angular resolution for fifth zone
#del6 = 1.0      # increment in angular resolution for outermost zone
#z1 = 11.0       # outer edge of innermost zone (degrees)
#z2 = 15.0       # outer edge of second zone
#z3 = 20.0       # outer edge of third zone
#z4 = 30.0       # outer edge of fourth zone
#z5 = 90.0       # outer edge of fifth zone
#azm = 0.5       # increment in azimuthal resolution (degrees)
 
# -- Mesh Paramters -- Lower Resolution (faster)
del1 = 0.001    # increment in angular resolution (degrees) for innermost zone
del2 = 0.01     # increment in angular resolution for second zone
del3 = 0.1      # increment in angular resolution for third zone
del4 = 0.2      # increment in angular resolution for fourth zone
del5 = 0.5      # increment in angular resolution for fifth zone
del6 = 1.0      # increment in angular resolution for outermost zone
z1 = 0.6        # outer edge of innermost zone (degrees)
z2 = 1.0        # outer edge of second zone
z3 = 2.0        # outer edge of third zone
z4 = 5.0        # outer edge of fourth zone
z5 = 10.0       # outer edge of fifth zone
azm = 1.0       # increment in azimuthal resolution (degrees)
 
# Optional: Additional string to include in output filenames (e.g. "_2019")
if (common_mesh == True):
    mtag = "commonMesh"
else:
    mtag = "stationMesh"
outstr = ("_" + mtag + "_" + pmod)

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

# Ensure that the Output Directories Exist
if (rank == 0):
    if not (os.path.isdir("../output/Convolution/")):
        os.makedirs("../output/Convolution/")
    if not (os.path.isdir("../output/Convolution/temp/")):
        os.makedirs("../output/Convolution/temp/")
    tempdir = "../output/Convolution/temp/"

    # Read Station File
    slat,slon,sta = read_station_file.main(sta_file)
 
    # Ensure that Station Locations are in Range 0-360
    neglon_idx = np.where(slon<0.)
    slon[neglon_idx] += 360.

    # Determine Number of Stations Read In
    if isinstance(slat,float) == True: # only 1 station
        numel = 1
    else:
        numel = len(slat)
 
    # Generate an Array of File Indices
    sta_idx = np.linspace(0,numel,num=numel,endpoint=False)
    np.random.shuffle(sta_idx)

else: # If I'm a worker, I know nothing yet about the data
    slat = slon = sta = numel = sta_idx = None

# Make Sure Everyone Has Reported Back Before Moving On
comm.Barrier()

# All Processors Get Certain Arrays and Parameters; Broadcast Them
sta          = comm.bcast(sta, root=0)
slat         = comm.bcast(slat, root=0)
slon         = comm.bcast(slon, root=0)
numel        = comm.bcast(numel, root=0)
sta_idx      = comm.bcast(sta_idx, root=0)
 
# MPI: Determine the Chunk Sizes for the Convolution
total_stations = len(slat)
nominal_load = total_stations // size # Floor Divide
# Final Chunk Might Be Different in Size Than the Nominal Load
if rank == size - 1:
    procN = total_stations - rank * nominal_load
else:
    procN = nominal_load

# Make some preparations that are common to all stations
if (rank == 0):  

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
    lslon[neglon_idx] += 360. 
 
    # Convert Start and End Dates to Datetimes
    if (time_series == True):
        frstdt = datetime.datetime(frst_date[0],frst_date[1],frst_date[2],frst_date[3],frst_date[4],frst_date[5])
        lastdt = datetime.datetime(last_date[0],last_date[1],last_date[2],last_date[3],last_date[4],last_date[5])

    # Check format of load files
    if not (loadfile_format == "nc"):
        if not (loadfile_format == "txt"):
            print(":: Error: Invalid format for load files. See scripts in the /GRDGEN/load_files/ folder. \
                Acceptable formats: netCDF, txt.")

    # Determine Number of Matching Load Files
    load_files = []
    if os.path.isdir(loadfile_directory):
        for mfile in os.listdir(loadfile_directory): # Filter by Load Directory
            if mfile.startswith(loadfile_prefix): # Filter by File Prefix
                if (time_series == True):
                    if (loadfile_format == "txt"):
                        mydt = datetime.datetime.strptime(mfile[-18:-4],'%Y%m%d%H%M%S') # Convert Filename String to Datetime
                    elif (loadfile_format == "nc"):
                        mydt = datetime.datetime.strptime(mfile[-17:-3],'%Y%m%d%H%M%S') # Convert Filename String to Datetime
                    else:
                        print(":: Error: Invalid format for load files. See scripts in the /GRDGEN/load_files/ folder. \
                            Acceptable formats: netCDF, txt.")
                    if ((mydt >= frstdt) & (mydt <= lastdt)): # Filter by Date Range
                        load_files.append(loadfile_directory + mfile) # Append File to List
                else:
                    load_files.append(loadfile_directory + mfile) # Append File to List
    else:
        sys.exit('Error: The loadfile directory does not exist. You may need to create it. \
            The /GRDGEN/load_files/ folder contains utility scripts to convert common models into \
            LoadDef-compatible formats, and will automatically create a loadfile directory.')

    # Test for Load Files
    if not load_files:
        sys.exit('Error: Could not find load files. You may need to generate them. \
            The /GRDGEN/load_files/ folder contains utility scripts to convert \
            common models into LoadDef-compatible formats.')

    # Sort the Filenames
    load_files = np.asarray(load_files)
    fidx = np.argsort(load_files)
    load_files = load_files[fidx]
    num_lfiles = len(load_files)
 
    # Initialize Arrays
    eamp = np.empty((numel,num_lfiles))
    epha = np.empty((numel,num_lfiles))
    namp = np.empty((numel,num_lfiles))
    npha = np.empty((numel,num_lfiles))
    vamp = np.empty((numel,num_lfiles))
    vpha = np.empty((numel,num_lfiles))

# If I'm a Worker, I Know Nothing About the Data
else:
    lslat = lslon = lsmask = load_files = None
    eamp = epha = namp = npha = vamp = vpha = None

# Make Sure Everyone Has Reported Back Before Moving On
comm.Barrier()

# Prepare the common mesh, if applicable
if (rank == 0):
    if (common_mesh == True): 

        ## Read in the common mesh
        print(':: Common Mesh True. Reading in ilat, ilon, iarea.')
        lcext = convmesh[-2::]
        if (lcext == 'xt'):
            ilat,ilon,unit_area = np.loadtxt(convmesh,usecols=(0,1,2),unpack=True)
            # convert from unit area to true area of the spherical patch in m^2
            iarea = np.multiply(unit_area, planet_radius**2) 
        elif (lcext == 'nc'):
            f = netCDF4.Dataset(convmesh)
            ilat = f.variables['midpoint_lat'][:]
            ilon = f.variables['midpoint_lon'][:]
            unit_area = f.variables['unit_area_patch'][:]
            f.close()
            # convert from unit area to true area of the spherical patch in m^2
            iarea = np.multiply(unit_area, planet_radius**2)

        ## Determine the Land-Sea Mask: Interpolate onto Mesh
        print(':: Common Mesh True. Applying Land-Sea Mask.')
        print(':: Number of Grid Points: %s | Size of LSMask: %s' %(str(len(ilat)), str(lsmask.shape)))
        lsmk = interpolate_lsmask.main(ilat,ilon,lslat,lslon,lsmask)
        print(':: Finished LSMask Interpolation.')

        ## For a common mesh, can already interpolate the load(s) onto the mesh, and also apply the land-sea mask.
        ## Prepare land-sea mask application
        if (lsmask_type == 2): 
            test_elements = np.where(lsmk == 0); test_elements = test_elements[0]
        elif (lsmask_type == 1): 
            test_elements = np.where(lsmk == 1); test_elements = test_elements[0]

        ## Loop through load file(s)
        full_files = []
        for hh in range(0,len(load_files)):

            ## Current load file
            cldfile = load_files[hh]

            ## Filename identifier
            str_components = cldfile.split('_')
            cext = str_components[-1]
            if (loadfile_format == "txt"):
                file_id = cext[0:-4]
            elif (loadfile_format == "nc"):
                file_id = cext[0:-3]
            else:
                print(':: Error. Invalid file format for load models. [load_convolution.py]')
                sys.exit()

            ## Read the File
            llat,llon,amp,pha,llat1dseq,llon1dseq,amp2darr,pha2darr = read_AmpPha.main(cldfile,loadfile_format,regular_grid=regular)
            ## Find Where Amplitude is NaN (if anywhere) and Set to Zero
            nanidx = np.isnan(amp); amp[nanidx] = 0.; pha[nanidx] = 0.
            ## Convert Amp/Pha Arrays to Real/Imag
            real = np.multiply(amp,np.cos(np.multiply(pha,pi/180.)))
            imag = np.multiply(amp,np.sin(np.multiply(pha,pi/180.)))
            
            ## Interpolate Load at Each Grid Point onto the Integration Mesh
            ic1,ic2   = interpolate_load.main(ilat,ilon,llat,llon,real,imag,regular)
            
            ## Multiply the Load Heights by the Load Density
            ic1 = np.multiply(ic1,ldens)
            ic2 = np.multiply(ic2,ldens)
            
            ## Enforce Mass Conservation, if Desired
            if (mass_cons == True):
                if (lsmask_type == 1): # For Oceans
                    print(':: Warning: Enforcing Mass Conservation Over Oceans.')
                    ic1_mc,ic2_mc = mass_conservation.main(ic1[lsmk==0],ic2[lsmk==0],iarea[lsmk==0])
                    ic1[lsmk==0] = ic1_mc
                    ic2[lsmk==0] = ic2_mc
                else: # For Land and Whole-Globe Models (like atmosphere and continental water)
                    print(':: Warning: Enforcing Mass Conservation Over Entire Globe.')
                    ic1,ic2 = mass_conservation.main(ic1,ic2,iarea)

            ## Apply Land-Sea Mask Based on LS Mask Database (LAND=1;OCEAN=0) 
            # If lsmask_type = 2, Set Oceans to Zero (retain land)
            # If lsmask_type = 1, Set Land to Zero (retain ocean)
            # Else, Do Nothing (retain full model)
            if (lsmask_type == 2):
                ic1[lsmk == 0] = 0.
                ic2[lsmk == 0] = 0. 
            elif (lsmask_type == 1):
                ic1[lsmk == 1] = 0.
                ic2[lsmk == 1] = 0.

            ## Write results to temporary netCDF files
            print(":: Writing netCDF-formatted temporary file for: ", cldfile)
            custom_file = (tempdir + "temp" + outstr + "_" + file_id + ".nc")
            full_files.append(custom_file)
            # Open new NetCDF file in "write" mode
            dataset = netCDF4.Dataset(custom_file,'w',format='NETCDF4_CLASSIC')
            # Define dimensions for variables
            num_pts = len(ic1)
            latitude = dataset.createDimension('latitude',num_pts)
            longitude = dataset.createDimension('longitude',num_pts)
            real = dataset.createDimension('real',num_pts)
            imag = dataset.createDimension('imag',num_pts)
            parea = dataset.createDimension('area',num_pts)
            # Create variables
            latitudes = dataset.createVariable('latitude',float,('latitude',))
            longitudes = dataset.createVariable('longitude',float,('longitude',))
            reals = dataset.createVariable('real',float,('real',))
            imags = dataset.createVariable('imag',float,('imag',))
            pareas = dataset.createVariable('area',float,('area',))
            # Add units
            latitudes.units = 'degree_north'
            longitudes.units = 'degree_east'
            reals.units = 'kg/m^2 (real part of load * load density)'
            imags.units = 'kg/m^2 (imag part of load * load density)'
            pareas.units = 'm^2 (unit area of patch * planet_radius^2)'
            # Assign data
            latitudes[:] = ilat
            longitudes[:] = ilon
            reals[:] = ic1
            imags[:] = ic2
            pareas[:] = iarea
            # Write Data to File
            dataset.close()
        
        ## Rename file list
        load_files = full_files.copy()

# Make Sure Everyone Has Reported Back Before Moving On
comm.Barrier()

## If Using a Common Mesh, Then Re-set the LoadFile Format to Indicate a Common Mesh is Used
if (common_mesh == True): 
    loadfile_format = "common"
  
# All Processors Get Certain Arrays and Parameters; Broadcast Them
lslat        = comm.bcast(lslat, root=0)
lslon        = comm.bcast(lslon, root=0)
lsmask       = comm.bcast(lsmask, root=0)
load_files   = comm.bcast(load_files, root=0)
eamp         = comm.bcast(eamp, root=0)
epha         = comm.bcast(epha, root=0)
namp         = comm.bcast(namp, root=0)
npha         = comm.bcast(npha, root=0)
vamp         = comm.bcast(vamp, root=0)
vpha         = comm.bcast(vpha, root=0) 

# Gather the Processor Workloads for All Processors
sendcounts = comm.gather(procN, root=0)
 
# Create a Data Type for the Convolution Results
cntype = MPI.DOUBLE.Create_contiguous(1)
cntype.Commit()

# Create a Data Type for Convolution Results for each Station and Load File
num_lfiles = len(load_files)
ltype = MPI.DOUBLE.Create_contiguous(num_lfiles)
ltype.Commit()
 
# Scatter the Station Locations (By Index)
d_sub = np.empty((procN,))
comm.Scatterv([sta_idx, (sendcounts, None), cntype], d_sub, root=0)

# Set up the arrays
eamp_sub = np.empty((len(d_sub),num_lfiles))
epha_sub = np.empty((len(d_sub),num_lfiles))
namp_sub = np.empty((len(d_sub),num_lfiles))
npha_sub = np.empty((len(d_sub),num_lfiles))
vamp_sub = np.empty((len(d_sub),num_lfiles))
vpha_sub = np.empty((len(d_sub),num_lfiles))

# Loop through the stations
for ii in range(0,len(d_sub)):

    # Current station
    current_sta = int(d_sub[ii]) # Index

    # Remove Index If Only 1 Station
    if (numel == 1): # only 1 station read in
        csta = sta
        clat = slat
        clon = slon
    else:
        csta = sta[current_sta]
        clat = slat[current_sta]
        clon = slon[current_sta]

    # If Rank is Main, Output Station Name
    try:
        csta = csta.decode()
    except:
        pass

    # Output File Name
    cnv_out = csta + "_" + rfm + "_" + loadfile_prefix + outstr + ".txt"

    # Status update
    print(':: Working on station: %s | Number: %6d of %6d | Rank: %6d' %(csta, (ii+1), len(d_sub), rank))
    
    # Compute Convolution for Current File
    eamp_sub[ii,:],epha_sub[ii,:],namp_sub[ii,:],npha_sub[ii,:],vamp_sub[ii,:],vpha_sub[ii,:] = load_convolution.main(\
        grn_file,norm_flag,load_files,loadfile_format,regular,lslat,lslon,lsmask,lsmask_type,clat,clon,csta,cnv_out,load_density=ldens,\
        delinc1=del1,delinc2=del2,delinc3=del3,delinc4=del4,delinc5=del5,delinc6=del6,izb=z1,z2b=z2,z3b=z3,z4b=z4,z5b=z5,azminc=azm)
  
# Gather Results
comm.Gatherv(eamp_sub, [eamp, (sendcounts, None), ltype], root=0)
comm.Gatherv(epha_sub, [epha, (sendcounts, None), ltype], root=0)
comm.Gatherv(namp_sub, [namp, (sendcounts, None), ltype], root=0)
comm.Gatherv(npha_sub, [npha, (sendcounts, None), ltype], root=0)
comm.Gatherv(vamp_sub, [vamp, (sendcounts, None), ltype], root=0)
comm.Gatherv(vpha_sub, [vpha, (sendcounts, None), ltype], root=0)

# Make Sure Everyone Has Reported Back Before Moving On
comm.Barrier()

# Free Data Type
cntype.Free()
ltype.Free()

# Re-organize Solutions
if (rank == 0):
    narr,nidx = np.unique(sta_idx,return_index=True)
    try:
        eamp = eamp[nidx,:]; namp = namp[nidx,:]; vamp = vamp[nidx,:]
        epha = epha[nidx,:]; npha = npha[nidx,:]; vpha = vpha[nidx,:]
    except: 
        eamp = eamp[nidx]; namp = namp[nidx]; vamp = vamp[nidx]
        epha = epha[nidx]; npha = npha[nidx]; vpha = vpha[nidx]
    #print('Up amplitude (rows = stations; cols = load models):')
    #print(vamp)
    #print('Up phase (rows = stations; cols = load models):')
    #print(vpha)

# Make Sure All Jobs Have Finished Before Continuing
comm.Barrier()

# Remove load files that are no longer needed
if (rank == 0):
    if (common_mesh == True): 
        for gg in range(0,len(load_files)): 
            cfile = load_files[gg]
            os.remove(cfile) 

# Make Sure All Jobs Have Finished Before Continuing
comm.Barrier()

# --------------------- END CODE --------------------------- #

