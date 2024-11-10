#!/usr/bin/env python

# *********************************************************************
# MAIN PROGRAM TO COMPUTE A DESIGN MATRIX TO INVERT FOR STRUCTURE --
#
# Copyright (c) 2022-2024: HILARY R. MARTENS
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
from scipy import interpolate
from LOADGF.utility import perturb_pmod
from LOADGF.LN import compute_love_numbers
from LOADGF.GF import compute_greens_functions
from CONVGF.CN import load_convolution
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
from utility.pmes import combine_stations
from CONVGF.utility import read_convolution_file

# --------------- SPECIFY USER INPUTS --------------------- #

# Full path to planet model text file
#     Planet model should be spherically symmetric, elastic,
#         non-rotating, and isotropic (SNREI)
#     Format: radius(km), vp(km/s), vs(km/s), density(g/cc)
#     If the file delimiter is not whitespace, then specify in
#         call to function.
pmod = "PREM"
planet_model = ("../input/Planet_Models/" + pmod + ".txt")

# Perturbation used for the forward-model runs
perturbation = np.log10(1.01)

# Regions perturbed in the forward-model runs
#   Note: The second-order Tikhonov regularization in StructSolv will only work properly if the layers stack on one another.
#         For example, the bottom radius of the top-most layer is the top radius of the next layer down. 
nodes = [[6351.,6371.],[6331.,6351.],[6311.,6331.]]

# Reference frame [Blewitt 2003]
rfm = "cm"

# Full Path to Load Directory and Prefix of Filename
loadfile_directory = ("../output/Grid_Files/nc/OTL/") 

# Prefix for the Load Files (Load Directory will be Searched for all Files Starting with this Prefix)
#  :: Note: For Load Files Organized by Date, the End of Filename Name Must be in the Format yyyymmddhhmnsc.txt
#  :: Note: If not organized by date, files may be organized by tidal harmonic, for example (i.e. a unique filename ending)
#  :: Note: Output names (within output files) will be determined by extension following last underscore character (e.g., date/harmonic/model)
loadfile_prefix = ("convgf_GOT410c") 

# LoadFile Format: ["nc", "txt"]
loadfile_format = "nc"

# Include imaginary component? For harmonic loads, such as tides, set to "True." Otherwise, for standard displacement data, set to "False."
inc_imag = True

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
#  Recommended: 1025-1035 kg/m^3 for oceanic loads (e.g., FES2014, ECCO2); 1 kg/m^3 for atmospheric loads (e.g. ECMWF); 1000 kg/m^3 for fresh water
ldens = 1030.0

# NEW OPTION: Provide a common geographic mesh?
# If True, must provide the full path to a mesh file (see: GRDGEN/common_mesh). 
# If False, a station-centered grid will be created within the functions called here. 
common_mesh = True
# Full Path to Grid File Containing Surface Mesh (for sampling the load Green's functions)
#  :: Format: latitude midpoints [float,degrees N], longitude midpoints [float,degrees E], unit area of each patch [float,dimensionless (need to multiply by r^2)]
meshfname = ("commonMesh_global_1.0_1.0_18.0_60.0_213.0_278.0_0.1_0.1_28.0_50.0_233.0_258.0_0.01_0.01_landmask")
convmesh = ("../output/Grid_Files/nc/commonMesh/" + meshfname + ".nc")

# Planet Radius (in meters; used for Greens function normalization)
planet_radius = 6371000.
  
# Ocean/Land Mask 
#  :: 0 = do not mask ocean or land (retain full model); 1 = mask out land (retain ocean); 2 = mask out oceans (retain land)
#  :: Recommended: 1 for oceanic; 2 for atmospheric
lsmask_type = 1

# Full Path to Land-Sea Mask File (May be Irregular and Sparse)
#  :: Format: Lat, Lon, Mask [0=ocean; 1=land]
lsmask_file = ("../input/Land_Sea/ETOPO1_Ice_g_gmt4_wADD.txt")

# Enforce mass conservation by removing a spatial mean from the load grid?
mass_cons = False

# Station/Grid-Point Location File (Lat, Lon, StationName)
sta_file = ("../input/Station_Locations/NOTA.txt")

# Overwrite old convolution files? (May be helpful if you want to change / add load files, but keep everything else the same)
overwrite = True

# Optional: Additional string to include in all output filenames (Love numbers, Green's functions, Convolution)
outstr = ("")

# Optional: Additional string to include in output filenames for the convolution (e.g. "_2022")
if (common_mesh == True):
    mtag = "commonMesh"
else:
    mtag = "stationMesh"
outstr_conv = ("_dens" + str(int(ldens)) + "_" + mtag)
 
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
    if not (os.path.isdir("../output/DesignMatrixStructure/")):
        os.makedirs("../output/DesignMatrixStructure/")
    if not (os.path.isdir("../output/CombineStations/")):
        os.makedirs("../output/CombineStations/")
    if not (os.path.isdir("../output/Love_Numbers/")):
        os.makedirs("../output/Love_Numbers/")
    if not (os.path.isdir("../output/Love_Numbers/LLN/")):
        os.makedirs("../output/Love_Numbers/LLN")
    if not (os.path.isdir("../output/Love_Numbers/PLN/")):
        os.makedirs("../output/Love_Numbers/PLN")
    if not (os.path.isdir("../output/Love_Numbers/STR/")):
        os.makedirs("../output/Love_Numbers/STR")
    if not (os.path.isdir("../output/Love_Numbers/SHR/")):
        os.makedirs("../output/Love_Numbers/SHR")
    if not (os.path.isdir("../output/Greens_Functions/")):
        os.makedirs("../output/Greens_Functions/")
    if not (os.path.isdir("../output/Planet_Models/")):
        os.makedirs("../output/Planet_Models/")
    if not (os.path.isdir("../output/Convolution/temp/")):
        os.makedirs("../output/Convolution/temp/")
tempdir = "../output/Convolution/temp/"

# Check format of load files
if not (loadfile_format == "nc"):
    if not (loadfile_format == "txt"):
        print(":: Error: Invalid format for load files. See scripts in the /GRDGEN/load_files/ folder. Acceptable formats: netCDF, txt.")

# Make sure all jobs have finished before continuing
comm.Barrier()

# ---------------- BEGIN PERTURB MODEL ---------------------- #

# Create a list of planetary models based on layers to perturb
pmodels = [] # Names of planetary models
lngfext = [] # Extensions for Love number and Green's function files
lnfiles = [] # Names of load Love number files
gffiles = [] # Names of load Green's function files
outdir_pmods = ("../output/Planet_Models/")
# Loop through nodes
for ee in range(0,len(nodes)):
    # Current radial range
    crad_range = nodes[ee]
    # RUN THE PERTURBATIONS
    perturb_pmod.main(planet_model,pmod,perturbation,crad_range,outdir_pmods,suffix=outstr)
    # Current output name (must match what "perturb_pmod" produces!)
    outname = (str('{:.4f}'.format(perturbation)) + "_" + str(crad_range[0]) + "_" + str(crad_range[1]) + outstr)
    # New model for mu
    mu_name = (pmod + "_mu_" + outname)
    fname_mu = (outdir_pmods + mu_name + ".txt")
    lngfext_mu = (mu_name + outstr + ".txt")
    ln_mu = ("../output/Love_Numbers/LLN/lln_" + lngfext_mu)
    gf_mu = ("../output/Greens_Functions/" + rfm + "_" + lngfext_mu)
    # New model for kappa
    kappa_name = (pmod + "_kappa_" + outname)
    fname_kappa = (outdir_pmods + kappa_name + ".txt")
    lngfext_kappa = (kappa_name + outstr + ".txt")
    ln_kappa = ("../output/Love_Numbers/LLN/lln_" + lngfext_kappa)
    gf_kappa = ("../output/Greens_Functions/" + rfm + "_" + lngfext_kappa)
    # New model for rho
    rho_name = (pmod + "_rho_" + outname)
    fname_rho = (outdir_pmods + rho_name + ".txt")
    lngfext_rho = (rho_name + outstr + ".txt")
    ln_rho = ("../output/Love_Numbers/LLN/lln_" + lngfext_rho)
    gf_rho = ("../output/Greens_Functions/" + rfm + "_" + lngfext_rho)
    # Append files to list
    pmodels.append(fname_mu)
    pmodels.append(fname_kappa)
    pmodels.append(fname_rho)
    lngfext.append(lngfext_mu)
    lngfext.append(lngfext_kappa)
    lngfext.append(lngfext_rho)
    lnfiles.append(ln_mu)
    lnfiles.append(ln_kappa)
    lnfiles.append(ln_rho)
    gffiles.append(gf_mu)
    gffiles.append(gf_kappa)
    gffiles.append(gf_rho)
# Append original model
pmodels.append(planet_model)
lngfext.append(pmod + outstr + ".txt")
lnfiles.append("../output/Love_Numbers/LLN/lln_" + pmod + outstr + ".txt")
gffiles.append("../output/Greens_Functions/" + rfm + "_" + pmod + outstr + ".txt")    

# ---------------- END PERTURB MODEL ----------------------- #
 
# ---------------- BEGIN LOVE NUMBERS ---------------------- #

# Loop through planetary models 
for bb in range(0,len(pmodels)): 

    # Current model
    cpmod = pmodels[bb]

    # Output filename
    file_ext = lngfext[bb]

    # Check if file already exists
    if (os.path.isfile(lnfiles[bb])):
        continue
    else: 
 
        # Compute the Love numbers (Load and Potential)
        if (rank == 0):
            # Compute Love Numbers
            ln_n,ln_h,ln_nl,ln_nk,ln_h_inf,ln_l_inf,ln_k_inf,ln_h_inf_p,ln_l_inf_p,ln_k_inf_p,\
                ln_hpot,ln_nlpot,ln_nkpot,ln_hstr,ln_nlstr,ln_nkstr,ln_hshr,ln_nlshr,ln_nkshr,\
                ln_planet_radius,ln_planet_mass,ln_sint,ln_Yload,ln_Ypot,ln_Ystr,ln_Yshr,\
                ln_lmda_surface,ln_mu_surface = \
                compute_love_numbers.main(cpmod,rank,comm,size,file_out=file_ext)
        # For Worker Ranks, Run the Code But Don't Return Any Variables
        else:
            # Workers Compute Love Numbers
            compute_love_numbers.main(cpmod,rank,comm,size,file_out=file_ext)
            # Workers Will Know Nothing About the Data Used to Compute the GFs
            ln_n = ln_h = ln_nl = ln_nk = ln_h_inf = ln_l_inf = ln_k_inf = ln_h_inf_p = ln_l_inf_p = ln_k_inf_p = None
            ln_planet_radius = ln_planet_mass = ln_Yload = ln_Ypot = ln_Ystr = ln_Yshr = None
            ln_hpot = ln_nlpot = ln_nkpot = ln_hstr = ln_nlstr = ln_nkstr = ln_hshr = None
            ln_nlshr = ln_nkshr = ln_sint = ln_lmda_surface = ln_mu_surface = None

# ----------------- END LOVE NUMBERS ----------------------- #

# -------------- BEGIN GREENS FUNCTIONS -------------------- #

# Make sure all jobs have finished before continuing
comm.Barrier()

# Set normalization flag
norm_flag  = False

# Loop through Love number files
for cc in range(0,len(lnfiles)):

    # Current Love number file
    lln_file = lnfiles[cc]

    # Output filename
    file_out = lngfext[cc]

    # Check if file already exists
    if (os.path.isfile(gffiles[cc])):
        continue
    else: 
  
        # Compute the Displacement Greens functions (For Load Love Numbers Only)
        if (rank == 0):
            u,v,u_norm,v_norm,u_cm,v_cm,u_norm_cm,v_norm_cm,u_cf,v_cf,u_norm_cf,v_norm_cf,gE,gE_norm,gE_cm,gE_cm_norm,\
                gE_cf,gE_cf_norm,tE,tE_norm,tE_cm,tE_cm_norm,tE_cf,tE_cf_norm,\
                e_tt,e_ll,e_rr,e_tt_norm,e_ll_norm,e_rr_norm,e_tt_cm,e_ll_cm,e_rr_cm,e_tt_cm_norm,e_ll_cm_norm,e_rr_cm_norm,\
                e_tt_cf,e_ll_cf,e_rr_cf,e_tt_cf_norm,e_ll_cf_norm,e_rr_cf_norm,gN,tN = \
                    compute_greens_functions.main(lln_file,rank,comm,size,grn_out=file_out)
        # For Worker Ranks, Run the Code But Don't Return Any Variables
        else:
            compute_greens_functions.main(lln_file,rank,comm,size,grn_out=file_out)

# -------------- END GREENS FUNCTIONS ---------------------- #

# ---------------- BEGIN CONVOLUTIONS ---------------------- #

# Ensure that the Output Directories Exist & Read in the Stations
if (rank == 0):

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

# File information
cndirectory = ("../output/Convolution/")
if (lsmask_type == 2):
    cnprefix = ("cn_LandOnly_")
elif (lsmask_type == 1):
    cnprefix = ("cn_OceanOnly_")
else:
    cnprefix = ("cn_LandAndOceans_")

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

            ## Name of file and check whether it already exists
            custom_file = (tempdir + "temp" + outstr_conv + outstr + "_" + file_id + ".nc")
            full_files.append(custom_file)
            if os.path.isfile(custom_file):
                print(':: File exists: ', custom_file, ' -- moving on.')
                continue

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
 
# Gather the Processor Workloads for All Processors
sendcounts = comm.gather(procN, root=0)

# Create a Data Type for the Convolution Results
cntype = MPI.DOUBLE.Create_contiguous(1)
cntype.Commit()

# Create a Data Type for Convolution Results for each Station and Load File
num_lfiles = len(load_files)
ltype = MPI.DOUBLE.Create_contiguous(num_lfiles)
ltype.Commit()

# Set up suffix names
cnsuffixes = []

# Loop through Green's function files
for dd in range(0,len(gffiles)):

    # Current Green's functions
    grn_file = gffiles[dd]

    # Current filename extension
    c_outstr = lngfext[dd]
    c_outstr_noext = c_outstr[0:-4]

    # Current suffix
    csuffix = (rfm + "_" + loadfile_prefix + "_" + c_outstr_noext + outstr_conv + ".txt")
    cnsuffixes.append(csuffix)

    # Scatter the Station Locations (By Index)
    d_sub = np.empty((procN,))
    comm.Scatterv([sta_idx, (sendcounts, None), cntype], d_sub, root=0)

    # No need to Set up the arrays here; no need to use variables passed back from convolution
    # We will just write out the files, and then read them in again later.

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
        cnv_out = (csta + "_" + csuffix)

        # Full file name
        cn_fullpath = (cndirectory + cnprefix + cnv_out)

        # Check if file already exists
        if (os.path.isfile(cn_fullpath)):

            if (overwrite == True): 
                os.remove(cn_fullpath) 

            else: 
                print(":: File already exists: " + cn_fullpath + ". Continuing...")
                continue

        # Status update
        print(':: Working on station: %s | Number: %6d of %6d | Rank: %6d' %(csta, (ii+1), len(d_sub), rank))

        # Compute Convolution for Current File
        eamp,epha,namp,npha,vamp,vpha = load_convolution.main(\
            grn_file,norm_flag,load_files,loadfile_format,regular,lslat,lslon,lsmask,lsmask_type,clat,clon,csta,cnv_out,load_density=ldens)
 
    # No need to gather the data from MPI processors; no need to use variables passed back from convolution
    # We will just write out the files, and then read them in again later.
  
# Free Data Type
cntype.Free()
ltype.Free()

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
 
# ----------------- END CONVOLUTIONS ----------------------- #

# -------------- BEGIN COMBINE STATIONS -------------------- #

# Only execute on main processor
if (rank == 0):
 
    # List of all combined filenames
    combined_filenames = []
 
    # Loop through model files
    for ff in range(0,len(cnsuffixes)):

        # Current convolution suffix
        ccnsuffix = cnsuffixes[ff]

        # Combine the stations into a single file
        outdir_csta = ("../output/CombineStations/")
        c_combined_filenames = combine_stations.main(cndirectory,cnprefix,ccnsuffix,output_directory=outdir_csta)

        # Append to list
        combined_filenames.append(c_combined_filenames)

# Make sure all jobs have finished before continuing
comm.Barrier()  

# -------------- END COMBINE STATIONS ---------------------- #

# ------------- BEGIN FINITE DIFFERENCE -------------------- #

# Only execute on main processor
if (rank == 0):
 
    # Take the difference between each perturbed displacement and the displacements predicted by the primary model (m0)
    #  :: d(Gm)/dm
    #  :: Separately for east, north, up at each station
    #  :: [Gm' - Gm0] / [m' - m0]

    # How many design matrices are we producing? (There will be one for each load model)
    main_files = combined_filenames[-1]
    main_files_out = []
 
    # Loop through main files
    for gg in range(0,len(main_files)):
 
        # Current main file
        mfile = main_files[gg]
    
        # Extract text of filename
        mfilevals = mfile.split('/')
        mfilename = mfilevals[-1]
        mfilename = mfilename[0:-4]

        # Read the main file
        sta,lat,lon,eamp,epha,namp,npha,vamp,vpha = read_convolution_file.main(mfile)
 
        # Convert from amplitude and phase to displacement
        if (inc_imag == False): 
            edisp = np.multiply(eamp,np.cos(np.multiply(epha,(np.pi/180.))))
            ndisp = np.multiply(namp,np.cos(np.multiply(npha,(np.pi/180.))))
            udisp = np.multiply(vamp,np.cos(np.multiply(vpha,(np.pi/180.))))
        # Convert Amp+Phase to Real+Imag
        elif (inc_imag == True): 
            ere = np.multiply(eamp,np.cos(np.multiply(epha,(np.pi/180.))))
            nre = np.multiply(namp,np.cos(np.multiply(npha,(np.pi/180.))))
            ure = np.multiply(vamp,np.cos(np.multiply(vpha,(np.pi/180.))))
            eim = np.multiply(eamp,np.sin(np.multiply(epha,(np.pi/180.))))
            nim = np.multiply(namp,np.sin(np.multiply(npha,(np.pi/180.))))
            uim = np.multiply(vamp,np.sin(np.multiply(vpha,(np.pi/180.))))
        else: 
            sys.exit(':: Error: Incorrect selection for whether to include imaginary components. Must be True or False.')

        # Export a simple text file of the original model
        f_out_main = ("startingmodel_" + mfilename + ".txt")
        f_file_main = ("../output/DesignMatrixStructure/" + f_out_main)
        main_files_out.append(f_file_main)
        temp_head = ("./temp_head_" + str(np.random.randint(500)) + ".txt")
        temp_body = ("./temp_body_" + str(np.random.randint(500)) + ".txt")
        # Prepare Data for Output (as Structured Array)
        if (inc_imag == False): 
            all_data = np.array(list(zip(sta,lat,lon,edisp,ndisp,udisp)), dtype=[('sta','U8'), \
                ('lat',float),('lon',float),('edisp',float),('ndisp',float),('udisp',float)])
            # Write Header Info to File
            hf = open(temp_head,'w')
            temp_str = 'Station  Lat(+N,deg)  Lon(+E,deg)  E-Disp(mm)  N-Disp(mm)  U-Disp(mm)  \n'
            hf.write(temp_str)
            hf.close()
            # Write Model Results to File
            np.savetxt(temp_body,all_data,fmt=["%s"]+["%.7f",]*5,delimiter="        ")
        else:
            all_data = np.array(list(zip(sta,lat,lon,ere,nre,ure,eim,nim,uim)), dtype=[('sta','U8'), \
                ('lat',float),('lon',float),('ere',float),('nre',float),('ure',float),('eim',float),('nim',float),('uim',float)])
            # Write Header Info to File
            hf = open(temp_head,'w')
            temp_str = 'Station  Lat(+N,deg)  Lon(+E,deg)  E-Disp-Re(mm)  N-Disp-Re(mm)  U-Disp-Re(mm)  E-Disp-Im(mm)  N-Disp-Im(mm)  U-Disp-Im(mm)   \n'
            hf.write(temp_str)
            hf.close()
            # Write Model Results to File
            np.savetxt(temp_body,all_data,fmt=["%s"]+["%.7f",]*8,delimiter="        ")
        # Combine Header and Body Files
        filenames_main = [temp_head, temp_body]
        with open(f_file_main,'w') as outfile:
            for fname in filenames_main:
                with open(fname) as infile:
                    outfile.write(infile.read())
        # Remove Header and Body Files
        os.remove(temp_head)
        os.remove(temp_body) 

        # Set up current design matrix
        if (inc_imag == False):
            rowdim = len(sta)*3 # Multiply by three for the three spatial components (e,n,u)
        else:
            rowdim = len(sta)*6 # Multiply by six for the three spatial components (e,n,u), and real & imaginary components for each
        coldim = len(combined_filenames)-1 # -1 so as not to include the main file (only the perturbations to structure; no. of depth ranges * 3 for mu,kappa,rho)
        desmat = np.zeros((rowdim,coldim)) 
        dmrows = np.empty((rowdim,),dtype='U10') # Assumes that station names are no more than 9 characters in length (with E, N, or U also appended)
        sclat = np.zeros((rowdim,))
        sclon = np.zeros((rowdim,))
        bottom_radius = np.zeros((coldim,))
        top_radius = np.zeros((coldim,))
        mat_param = np.empty((coldim,),dtype='U10')

        # Loop through other files that correspond to this main file (perturbations to structure)
        for hh in range(0,len(combined_filenames)-1): # -1 so as not to include the main file

            # Current file with material perturbation
            cpfiles = combined_filenames[hh]
            cpfile = cpfiles[gg]

            # Current depth bottom, depth top, and material parameter
            clngfext = lngfext[hh] # information on current model parameter
            clngfext_rmtxt = clngfext[0:-4] # remove the ".txt" extension
            perturbvars = clngfext_rmtxt.split('_')
            bottom_radius[hh] = perturbvars[3]
            top_radius[hh] = perturbvars[4]
            mat_param[hh] = perturbvars[1]

            # Read the current perturbed file
            sta1,lat1,lon1,eamp1,epha1,namp1,npha1,vamp1,vpha1 = read_convolution_file.main(cpfile)

            # Convert from amplitude and phase to displacement
            if (inc_imag == False):
                edisp1 = np.multiply(eamp1,np.cos(np.multiply(epha1,(np.pi/180.))))
                ndisp1 = np.multiply(namp1,np.cos(np.multiply(npha1,(np.pi/180.))))
                udisp1 = np.multiply(vamp1,np.cos(np.multiply(vpha1,(np.pi/180.))))
            # Convert Amp+Phase to Real+Imag
            else:
                ere1 = np.multiply(eamp1,np.cos(np.multiply(epha1,(np.pi/180.))))
                nre1 = np.multiply(namp1,np.cos(np.multiply(npha1,(np.pi/180.))))
                ure1 = np.multiply(vamp1,np.cos(np.multiply(vpha1,(np.pi/180.))))
                eim1 = np.multiply(eamp1,np.sin(np.multiply(epha1,(np.pi/180.))))
                nim1 = np.multiply(namp1,np.sin(np.multiply(npha1,(np.pi/180.))))
                uim1 = np.multiply(vamp1,np.sin(np.multiply(vpha1,(np.pi/180.))))

            # Subtract displacements from those displacements in the main file
            # And then divide by the perturbation. We want: dG(m)/dm, where dm=m'-m0
            # In log space, m' = log10(m'_linear) and m0 = log10(m0_linear).
            # To perturb the model parameters, we have: m' = m0 + "perturbation".
            # Thus, perturbation = m' - m0, and we want to compute: dG(m)/perturbation.
            # Hence, here, we compute the difference in displacement and divide by the perturbation.
            if (inc_imag == False):
                edisp_diff = np.divide(np.subtract(edisp1,edisp),perturbation)
                ndisp_diff = np.divide(np.subtract(ndisp1,ndisp),perturbation)
                udisp_diff = np.divide(np.subtract(udisp1,udisp),perturbation)
            else:
                ere_diff = np.divide(np.subtract(ere1,ere),perturbation)
                nre_diff = np.divide(np.subtract(nre1,nre),perturbation)
                ure_diff = np.divide(np.subtract(ure1,ure),perturbation)
                eim_diff = np.divide(np.subtract(eim1,eim),perturbation)
                nim_diff = np.divide(np.subtract(nim1,nim),perturbation)
                uim_diff = np.divide(np.subtract(uim1,uim),perturbation)

            # Loop through stations
            for jj in range(0,len(sta1)): 
 
                # Fill in Design Matrix
                if (inc_imag == False): 
                    idxe = (jj*3)+0
                    idxn = (jj*3)+1
                    idxu = (jj*3)+2
                    desmat[idxe,hh] = edisp_diff[jj]
                    desmat[idxn,hh] = ndisp_diff[jj]
                    desmat[idxu,hh] = udisp_diff[jj]
                    dmrows[idxe] = (sta1[jj] + 'E')
                    dmrows[idxn] = (sta1[jj] + 'N')
                    dmrows[idxu] = (sta1[jj] + 'U')
                    sclat[idxe] = lat1[jj]
                    sclat[idxn] = lat1[jj]
                    sclat[idxu] = lat1[jj]
                    sclon[idxe] = lon1[jj]
                    sclon[idxn] = lon1[jj]
                    sclon[idxu] = lon1[jj]
                else:
                    idxere = (jj*6)+0
                    idxnre = (jj*6)+1
                    idxure = (jj*6)+2
                    idxeim = (jj*6)+3
                    idxnim = (jj*6)+4
                    idxuim = (jj*6)+5
                    desmat[idxere,hh] = ere_diff[jj]
                    desmat[idxnre,hh] = nre_diff[jj]
                    desmat[idxure,hh] = ure_diff[jj]
                    desmat[idxeim,hh] = eim_diff[jj]
                    desmat[idxnim,hh] = nim_diff[jj]
                    desmat[idxuim,hh] = uim_diff[jj]
                    dmrows[idxere] = (sta1[jj] + 'Ere')
                    dmrows[idxnre] = (sta1[jj] + 'Nre')
                    dmrows[idxure] = (sta1[jj] + 'Ure')
                    dmrows[idxeim] = (sta1[jj] + 'Eim')
                    dmrows[idxnim] = (sta1[jj] + 'Nim')
                    dmrows[idxuim] = (sta1[jj] + 'Uim')
                    sclat[idxere] = lat1[jj]
                    sclat[idxnre] = lat1[jj]
                    sclat[idxure] = lat1[jj]
                    sclon[idxere] = lon1[jj]
                    sclon[idxnre] = lon1[jj]
                    sclon[idxure] = lon1[jj]
                    sclat[idxeim] = lat1[jj]
                    sclat[idxnim] = lat1[jj]
                    sclat[idxuim] = lat1[jj]
                    sclon[idxeim] = lon1[jj]
                    sclon[idxnim] = lon1[jj]
                    sclon[idxuim] = lon1[jj]

        # Write Design Matrix to File
        print(":: ")
        print(":: ")
        print(":: Writing netCDF-formatted file.")
        f_out = ("designmatrix_" + mfilename + ".nc")
        f_file = ("../output/DesignMatrixStructure/" + f_out)
        # Check if file already exists; if so, delete existing file
        if (os.path.isfile(f_file)):
            os.remove(f_file)
        # Open new NetCDF file in "write" mode
        dataset = netCDF4.Dataset(f_file,'w',format='NETCDF4_CLASSIC')
        # Define dimensions for variables
        desmat_shape = desmat.shape
        num_rows = desmat_shape[0]
        num_cols = desmat_shape[1]
        nstacomp = dataset.createDimension('nstacomp',num_rows)
        nstructure = dataset.createDimension('nstructure',num_cols)
        nchars = dataset.createDimension('nchars',10)
        # Create variables
        sta_comp_id = dataset.createVariable('sta_comp_id','S1',('nstacomp','nchars'))
        design_matrix = dataset.createVariable('design_matrix',float,('nstacomp','nstructure'))
        sta_comp_lat = dataset.createVariable('sta_comp_lat',float,('nstacomp',))
        sta_comp_lon = dataset.createVariable('sta_comp_lon',float,('nstacomp',))
        perturb_radius_bottom = dataset.createVariable('perturb_radius_bottom',float,('nstructure',))
        perturb_radius_top = dataset.createVariable('perturb_radius_top',float,('nstructure',))
        perturb_param = dataset.createVariable('perturb_param','S1',('nstructure','nchars'))
        # Add units
        sta_comp_id.units = 'string'
        if (inc_imag == False): 
            sta_comp_id.long_name = 'station_component_id'
        else: 
            sta_comp_id.long_name = 'station_component_RealImaginary_id'
        design_matrix.units = 'mm'
        design_matrix.long_name = 'displacement_mm'
        sta_comp_lat.units = 'degrees_north'
        sta_comp_lat.long_name = 'station_latitude'
        sta_comp_lon.units = 'degrees_east'
        sta_comp_lon.long_name = 'station_longitude'
        perturb_radius_bottom.units = 'km'
        perturb_radius_bottom.long_name = 'bottom_of_perturbed_layer'
        perturb_radius_top.units = 'km'
        perturb_radius_top.long_name = 'top_of_perturbed_layer'
        perturb_param.units = 'string'
        perturb_param.long_name = 'material_parameter_perturbed'
        # Assign data
        #  https://unidata.github.io/netcdf4-python/ (see "Dealing with Strings")
        #  sta_comp_id[:] = netCDF4.stringtochar(np.array(dmrows,dtype='S10'))
        sta_comp_id._Encoding = 'ascii'
        sta_comp_id[:] = np.array(dmrows,dtype='S10')
        design_matrix[:,:] = desmat
        sta_comp_lat[:] = sclat
        sta_comp_lon[:] = sclon
        perturb_radius_bottom[:] = bottom_radius
        perturb_radius_top[:] = top_radius
        perturb_param._Encoding = 'ascii'
        perturb_param[:] = np.array(mat_param,dtype='S10')
    
        # Write Data to File
        dataset.close()

        # Print the output filename
        print(f_file)

        # Read the netCDF file as a test
        f = netCDF4.Dataset(f_file)
        #print(f.variables)
        sta_comp_ids = f.variables['sta_comp_id'][:]
        design_matrix = f.variables['design_matrix'][:]
        sta_comp_lat = f.variables['sta_comp_lat'][:]
        sta_comp_lon = f.variables['sta_comp_lon'][:]
        perturb_radius_bottom = f.variables['perturb_radius_bottom'][:]
        perturb_radius_top = f.variables['perturb_radius_top'][:]
        perturb_param = f.variables['perturb_param'][:]
        f.close()

    # Remind users that they will also need the original forward models when they run the inversion:
    print(':: ')
    print(':: ')
    print(':: Reminder: You will also need the original forward model when running the inversion. [d-(Gm0)] = [d(Gm)/dm]*[dm]')
    print('::   (Gm0) represents the original forward model. [d-(Gm0)] represents the residual vector between GPS data and the original forward model')
    print('::   [d(Gm)/dm] represents the perturbations to the surface displacements with a perturbation to each model parameter.')
    print('::      It is the design matrix computed here. The default perturbation is 1%.')
    print('::   [dm] represents the model vector to be solved for in the inversion.')
    print('::      It is the perturbation to each model parameter required to best fit the residual data.')
    print(':: The original forward model(s) are: ')
    print(main_files)
    print(':: And the original forward model(s) recast into real and imaginary components are: ')
    print(main_files_out)
    print(':: ')
    print(':: ')
    print(':: Reminder: The planetary model that you have used to compute the design matrix and starting model: ')
    print(planet_model)
    print(':: ')
    print(':: ')

# -------------- END FINITE DIFFERENCE --------------------- #

# --------------------- END CODE --------------------------- #

