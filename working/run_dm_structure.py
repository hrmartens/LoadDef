#!/usr/bin/env python

# *********************************************************************
# MAIN PROGRAM TO COMPUTE A DESIGN MATRIX TO INVERT FOR STRUCTURE --
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
from LOADGF.utility import perturb_pmod
from LOADGF.LN import compute_love_numbers
from LOADGF.GF import compute_greens_functions
from CONVGF.CN import load_convolution
from CONVGF.utility import read_station_file
from CONVGF.utility import read_lsmask
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
  
# Ocean/Land Mask 
#  :: 0 = do not mask ocean or land (retain full model); 1 = mask out land (retain ocean); 2 = mask out oceans (retain land)
#  :: Recommended: 1 for oceanic; 2 for atmospheric
lsmask_type = 1

# Full Path to Land-Sea Mask File (May be Irregular and Sparse)
#  :: Format: Lat, Lon, Mask [0=ocean; 1=land]
lsmask_file = ("../input/Land_Sea/ETOPO1_Ice_g_gmt4_wADD.txt")

# Station/Grid-Point Location File (Lat, Lon, StationName)
sta_file = ("../input/Station_Locations/NOTA.txt")

# Optional: Additional string to include in all output filenames (Love numbers, Green's functions, Convolution)
outstr = ("")

# Optional: Additional string to include in output filenames for the convolution (e.g. "_2022")
outstr_conv = ("_dens" + str(int(ldens)) + "_2022")
 
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
    perturb_pmod.main(planet_model,pmod,perturbation,crad_range,outdir_pmods)
    # Current output name (must match what "perturb_pmod" produces!)
    outname = (str('{:.4f}'.format(perturbation)) + "_" + str(crad_range[0]) + "_" + str(crad_range[1]))
    # New model for mu
    mu_name = (pmod + "_mu_" + outname + outstr)
    fname_mu = (outdir_pmods + mu_name + ".txt")
    lngfext_mu = (mu_name + ".txt")
    ln_mu = ("../output/Love_Numbers/LLN/lln_" + lngfext_mu)
    gf_mu = ("../output/Greens_Functions/" + rfm + "_" + lngfext_mu)
    # New model for kappa
    kappa_name = (pmod + "_kappa_" + outname + outstr)
    fname_kappa = (outdir_pmods + kappa_name + ".txt")
    lngfext_kappa = (kappa_name + ".txt")
    ln_kappa = ("../output/Love_Numbers/LLN/lln_" + lngfext_kappa)
    gf_kappa = ("../output/Greens_Functions/" + rfm + "_" + lngfext_kappa)
    # New model for rho
    rho_name = (pmod + "_rho_" + outname + outstr)
    fname_rho = (outdir_pmods + rho_name + ".txt")
    lngfext_rho = (rho_name + ".txt")
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
    if (os.path.isfile(gffiles[bb])):
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

# File information
cndirectory = ("../output/Convolution/")
if (lsmask_type == 2):
    cnprefix = ("cn_LandOnly_")
elif (lsmask_type == 1):
    cnprefix = ("cn_OceanOnly_")
else:
    cnprefix = ("cn_LandAndOceans_")

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
print(':: Finished Reading in LSMask.')

# Ensure that Land-Sea Mask Longitudes are in Range 0-360
neglon_idx = np.where(lslon<0.)
 
# Read Station & Date Range File
lat,lon,sta = read_station_file.main(sta_file)

# Determine Number of Stations Read In
if isinstance(lat,float) == True: # only 1 station
    numel = 1
else:
    numel = len(lat)

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
            print(':: No need to decode station.')
        if (rank == 0):
            print(':: Starting on Station: ' + my_sta)

        # Output File Name
        cnv_out = (my_sta + "_" + csuffix)

        # Full file name
        cn_fullpath = (cndirectory + cnprefix + cnv_out)

        # Check if file already exists
        if (os.path.isfile(cn_fullpath)):
            print(":: File already exists: " + cn_fullpath + ". Continuing...")
            continue
        else:

            # Convert Start and End Dates to Datetimes
            if (time_series == True):
                frstdt = datetime.datetime(frst_date[0],frst_date[1],frst_date[2],frst_date[3],frst_date[4],frst_date[5])
                lastdt = datetime.datetime(last_date[0],last_date[1],last_date[2],last_date[3],last_date[4],last_date[5])

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
                                print(":: Error: Invalid format for load files. See scripts in the /GRDGEN/load_files/ folder. Acceptable formats: netCDF, txt.")
                            if ((mydt >= frstdt) & (mydt <= lastdt)): # Filter by Date Range
                                load_files.append(loadfile_directory + mfile) # Append File to List
                        else:
                            load_files.append(loadfile_directory + mfile) # Append File to List
            else:
                sys.exit('Error: The loadfile directory does not exist. You may need to create it. The /GRDGEN/load_files/ folder contains utility scripts to convert common models into LoadDef-compatible formats, and will automatically create a loadfile directory.')

            # Test for Load Files
            if not load_files:
                sys.exit('Error: Could not find load files. You may need to generate them. The /GRDGEN/load_files/ folder contains utility scripts to convert common models into LoadDef-compatible formats.')

            # Sort the Filenames
            load_files = np.asarray(load_files)
            fidx = np.argsort(load_files)
            load_files = load_files[fidx]

            # Set Lat/Lon/Name for Current Station
            slat = my_lat
            slon = my_lon
            sname = my_sta

            # Determine the Chunk Sizes for the Convolution
            total_files = len(load_files)
            nominal_load = total_files // size # Floor Divide
            # Final Chunk Might Be Different in Size Than the Nominal Load
            if rank == size - 1:
                procN = total_files - rank * nominal_load
            else:
                procN = nominal_load

            # Perform the Convolution for Each Station
            if (rank == 0):
                eamp,epha,namp,npha,vamp,vpha = load_convolution.main(grn_file,norm_flag,load_files,regular,lslat,lslon,lsmask,\
                    slat,slon,sname,cnv_out,lsmask_type,loadfile_format,rank,procN,comm,load_density=ldens)
            # For Worker Ranks, Run the Code But Don't Return Any Variables
            else:
                load_convolution.main(grn_file,norm_flag,load_files,regular,lslat,lslon,lsmask,\
                    slat,slon,sname,cnv_out,lsmask_type,loadfile_format,rank,procN,comm,load_density=ldens)

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
            if (inc_imag == False):
                edisp_diff = np.subtract(edisp1,edisp)
                ndisp_diff = np.subtract(ndisp1,ndisp)
                udisp_diff = np.subtract(udisp1,udisp)
            else:
                ere_diff = np.subtract(ere1,ere)
                nre_diff = np.subtract(nre1,nre)
                ure_diff = np.subtract(ure1,ure)
                eim_diff = np.subtract(eim1,eim)
                nim_diff = np.subtract(nim1,nim)
                uim_diff = np.subtract(uim1,uim)

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

# -------------- END FINITE DIFFERENCE --------------------- #

# --------------------- END CODE --------------------------- #

