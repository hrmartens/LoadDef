# *********************************************************************
# FUNCTION TO COMPUTE LOAD-INDUCED SURFACE DISPLACEMENTS
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

# Import Python Modules
from __future__ import print_function
from mpi4py import MPI
import numpy as np
import datetime
import scipy as sc
import netCDF4
from scipy import interpolate
from CONVGF.utility import read_greens_fcn_file
from CONVGF.utility import read_greens_fcn_file_norm
from CONVGF.utility import normalize_greens_fcns
from CONVGF.utility import read_cMesh
from CONVGF.utility import read_lsmask
from CONVGF.CN import compute_specific_greens_fcns
from CONVGF.CN import convolve_global_grid
from CONVGF.CN import generate_integration_mesh
from CONVGF.CN import intmesh2geogcoords
from CONVGF.CN import integrate_greens_fcns
from CONVGF.CN import interpolate_load
from CONVGF.CN import compute_angularDist_azimuth
from CONVGF.CN import coef2amppha
from CONVGF.CN import perform_convolution
from CONVGF.CN import interpolate_lsmask
import sys
import os
from math import pi

"""
Compute predicted surface displacements caused by surface mass loading by convolving 
displacement load Green's functions with a model for the surface mass load. 

Parameters
----------
load_density : Density of the surface mass load (kg/m^3)
    Default is 1030.0

rad : Mean Earth radius (m) 
    Default is 6371000.

# -- Mesh Paramters --
delinc1 : angular distance (degrees) increment for inner zone
    Default is 0.0002

delinc2 : angular distance (degrees) increment for zone 2
    Default is 0.001

delinc3 : angluar distance (degrees) increment for zone 3
    Default is 0.01

delinc4 : angluar distance (degrees) increment for zone 4
    Default is 0.1

delinc5 : angluar distance (degrees) increment for zone 5
    Default is 0.5

delinc6 : angular distance (degrees) increment for outer zone
    Default is 1.0

izb     : inner zone boundary (degrees; 0<izb<z2b)
    Default is 0.02

z2b     : zone 2 boundary (degrees; izb<z2b<z3b)
    Default is 0.1

z3b     : zone 3 boundary (degrees; z2b<z3b<z4b)
    Default is 1.0

z4b     : zone 4 boundary (degrees; z3b<z4b<z5b)
    Default is 10.0

z5b     : zone 5 boundary (degrees; z4b<z5b<180)
    Default is 90.0

azminc  : azimuthal increment # NOTE: Maybe should match azminc with delinc5 (i.e., keep azminc consistent with theta increment at 90 degrees from station,
                              #       where azimuth and theta increments are consistent in horizontal distance along planet's surface)
                              #       :: azminc*sin(theta) ~ delinc
    Default is 0.5 

mass_cons  :  option to enforce mass conservation by removing the spatial mean from the load grid
    Default is False

"""

def main(grn_file,norm_flag,load_files,loadfile_format,regular,lslat,lslon,lsmask,lsmask_type,slat,slon,stname,cnv_out,\
    load_density=1030.0,rad=6371000.,delinc1=0.0002,delinc2=0.001,delinc3=0.01,delinc4=0.1,delinc5=0.5,delinc6=1.0,\
    izb=0.02,z2b=0.1,z3b=1.0,z4b=10.0,z5b=90.0,azminc=0.5,mass_cons=False):

    # Determine Number of Load Files
    if isinstance(load_files,float) == True:
        numel = 1
    else: 
        numel = len(load_files)

    # Print Number of Files Read In
    display = ':: Number of Files Read In: ' + repr(numel)
    #print(display)

    # Ensure that Station Location is in Range 0-360
    if (slon < 0.):
        slon += 360.

    # Determine whether load file was supplied as a list of cells, or as traditional load files
    if (loadfile_format == "bbox"): # list of cells
        # Ensure only one file is read in for this format
        if (numel > 1):
            sys.exit(":: Error: For load files in 'bbox' format (i.e., a list of bounding boxes), only one file can be read in at a time. [load_convolution.py]")
        # Read in the file
        loadgrid = load_files[0]
        lcext = loadgrid[-2::]
        if (lcext == 'xt'):
            file_ids = np.loadtxt(loadgrid,usecols=(4,),unpack=True,dtype='U')
            southlat,northlat,westlon,eastlon = np.loadtxt(loadgrid,usecols=(0,1,2,3),unpack=True)
        elif (lcext == 'nc'):
            f = netCDF4.Dataset(loadgrid)
            file_ids = f.variables['cell_ids'][:]
            southlat = f.variables['slatitude'][:]
            northlat = f.variables['nlatitude'][:]
            westlon = f.variables['wlongitude'][:]
            eastlon = f.variables['elongitude'][:]
            f.close()
        # Ensure that Bounding Box Longitudes are in Range 0-360
        for yy in range(0,len(westlon)):
            if (westlon[yy] < 0.):
                westlon[yy] += 360.
            if (eastlon[yy] < 0.):
                eastlon[yy] += 360.
    elif (loadfile_format == "common"): # common mesh format
        # Create Array of Filename Extensions
        file_ids = []
        for qq in range(0,numel):
            mfile = load_files[qq]
            str_components = mfile.split('_')
            cext = str_components[-1]
            file_ids.append(cext[0:-3])
    else:
        # Create Array of Filename Extensions
        file_ids = []
        for qq in range(0,numel):
            mfile = load_files[qq]
            str_components = mfile.split('_')
            cext = str_components[-1]
            if (loadfile_format == "txt"):
                file_ids.append(cext[0:-4]) 
            elif (loadfile_format == "nc"):
                file_ids.append(cext[0:-3])
            else:
                print(':: Error. Invalid file format for load models. [load_convolution.py]')
                sys.exit()

    # Initialize Arrays
    eamp = np.empty((len(file_ids),))
    epha = np.empty((len(file_ids),))
    namp = np.empty((len(file_ids),))
    npha = np.empty((len(file_ids),))
    vamp = np.empty((len(file_ids),))
    vpha = np.empty((len(file_ids),))

    if (loadfile_format == "common"): # common geographic mesh

        # Read in the common mesh with the load and land-sea mask already applied
        # Unfortunately, need to read one file in outside the file loop; can revisit and optimize this later...
        # One option would be to pass ilat, ilon, and iarea into "load_convolution" via other variables that are not used for the common mesh (e.g., ls mask parameters)
        ilat, ilon, ic1, ic2, iarea = read_cMesh.main(load_files[0]) 

        # Read in the Green's Functions
        if norm_flag == True:
            theta,u,v,unormFarrell,vnormFarrell = read_greens_fcn_file_norm.main(grn_file,rad)
        else:
            theta,u,v,unormFarrell,vnormFarrell = read_greens_fcn_file.main(grn_file)

        # Normalize Green's According to Farrell Convention
        nfactor = 1E12*rad
        unorm = np.multiply(u,theta) * nfactor
        vnorm = np.multiply(v,theta) * nfactor

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

        # Un-normalize
        try: 
            uint = np.divide(uint,delta) / nfactor
            vint = np.divide(vint,delta) / nfactor
        except:
            print(':: Warning: Encountered an angle at or near zero; setting integrated LGFs to zero. [load_convolution.py]')
            uint = 0.
            vint = 0.

        # Compute Greens Functions Specific to Receiver and Grid (Geographic Coordinates)
        ur,ue,un = compute_specific_greens_fcns.main(haz,uint,vint)

        # Set Other Parameters to None, which are not needed for the common mesh
        lsmk = None 
 
    else: # standard station-centered grid

        # Generate Integration Mesh
        print(':: Generating the Integration Mesh. Please Wait...')
        gldel,glazm,ldel,lazm,unit_area = generate_integration_mesh.main(delinc1,delinc2, \
            delinc3,delinc4,delinc5,delinc6,izb,z2b,z3b,z4b,z5b,azminc)
 
        # Read Greens Function File
        if norm_flag == True:
            theta,u,v,unormFarrell,vnormFarrell = read_greens_fcn_file_norm.main(grn_file,rad)
        else:
            theta,u,v,unormFarrell,vnormFarrell = read_greens_fcn_file.main(grn_file)

        # Normalize Greens Functions (Agnew Convention)
        unorm,vnorm = normalize_greens_fcns.main(theta,u,v,rad)

        # Interpolate Greens Functions
        tck_gfu = interpolate.splrep(theta,unorm,k=3)
        tck_gfv = interpolate.splrep(theta,vnorm,k=3)

        # Integrate Greens Functions
        uint,vint = integrate_greens_fcns.main(gldel,glazm,ldel,lazm,tck_gfu,tck_gfv)

        # Compute Geographic Coordinates of Integration Mesh Cell Midpoints
        ilat,ilon,iarea = intmesh2geogcoords.main(slat,slon,ldel,lazm,unit_area)

        # Compute Angular Distance and Azimuth at Receiver Due to Load
        delta,haz = compute_angularDist_azimuth.main(slat,slon,ilat,ilon)

        # Compute Greens Functions Specific to Receiver and Grid (Geographic Coordinates)
        ur,ue,un = compute_specific_greens_fcns.main(haz,uint,vint)

        # Determine the Land-Sea Mask: Interpolate onto Mesh
        print(':: Interpolating the Land-Sea Mask. Please Wait...')
        print(':: Number of Grid Points: %s | Size of LSMask: %s' %(str(len(ilat)), str(lsmask.shape)))
        lsmk = interpolate_lsmask.main(ilat,ilon,lslat,lslon,lsmask)
        print(':: Finished LSMask Interpolation.')

        # Real and imaginary loading components are not yet defined (to be computed later)
        ic1 = ic2 = None

    # Loop through load models
    for ii in range(0,len(file_ids)):
        if (loadfile_format == "bbox"):
            csouthlat = southlat[ii]
            cnorthlat = northlat[ii]
            cwestlon = westlon[ii]
            ceastlon = eastlon[ii]
            mfile = [csouthlat,cnorthlat,cwestlon,ceastlon]
            file_id = file_ids[ii] # File Extension
            print(':: Working on Cell: %s | Station: %s | Number: %6d of %6d' %(file_id, stname, (ii+1), len(file_ids)))
        else: 
            mfile = load_files[ii] # Vector-Format 
            file_id = file_ids[ii] # File Extension
            print(':: Working on File: %s | Station: %s | Number: %6d of %6d' %(file_id, stname, (ii+1), len(file_ids)))
            # Check if Loadfile Exists
            if not (os.path.isfile(mfile)): # file does not exist
                eamp[ii] = np.nan
                epha[ii] = np.nan
                namp[ii] = np.nan
                npha[ii] = np.nan
                vamp[ii] = np.nan
                vpha[ii] = np.nan
                continue
        # Compute Convolution for Current File
        eamp[ii],epha[ii],namp[ii],npha[ii],vamp[ii],vpha[ii] = perform_convolution.main(\
            mfile,loadfile_format,ur,ue,un,load_density,ilat,ilon,iarea,lsmk,lsmask_type,regular,mass_cons,stname)

    # Prepare Output Files
    if (lsmask_type == 2):
        cnv_file = ("../output/Convolution/cn_LandOnly_" + cnv_out)
    elif (lsmask_type == 1):
        cnv_file = ("../output/Convolution/cn_OceanOnly_" + cnv_out)
    else:
        cnv_file = ("../output/Convolution/cn_LandAndOceans_" + cnv_out)
    cnv_head = ("../output/Convolution/"+str(np.random.randint(5000))+"cn_head.txt")
    cnv_body = ("../output/Convolution/"+str(np.random.randint(5000))+"cn_body.txt")
 
    # Prepare Data for Output (Create a Structured Array)
    slat_arr = np.ones((len(eamp),)) * slat
    slon_arr = np.ones((len(eamp),)) * slon
    if (loadfile_format == "bbox"):
        all_cnv_data = np.array(list(zip(file_ids,slat_arr,slon_arr,eamp,epha,namp,npha,vamp,vpha,southlat,northlat,westlon,eastlon)), dtype=[('file_ids','U25'), \
            ('slat_arr',float),('slon_arr',float),('eamp',float),('epha',float),('namp',float),('npha',float), \
            ('vamp',float),('vpha',float),('southlat',float),('northlat',float),('westlon',float),('eastlon',float)])
    else:
        all_cnv_data = np.array(list(zip(file_ids,slat_arr,slon_arr,eamp,epha,namp,npha,vamp,vpha)), dtype=[('file_ids','U25'), \
            ('slat_arr',float),('slon_arr',float),('eamp',float),('epha',float),('namp',float),('npha',float), \
            ('vamp',float),('vpha',float)])

    # Write Header Info to File
    hf = open(cnv_head,'w')
    if (loadfile_format == "bbox"):
        cnv_str = 'Extension/Epoch  Lat(+N,deg)  Lon(+E,deg)  E-Amp(mm)  E-Pha(deg)  N-Amp(mm)  N-Pha(deg)  V-Amp(mm)  V-Pha(deg)  South-Lat(deg)  North-Lat(deg)  West-Lon(deg)  East-Lon(deg) \n'
    else:
        cnv_str = 'Extension/Epoch  Lat(+N,deg)  Lon(+E,deg)  E-Amp(mm)  E-Pha(deg)  N-Amp(mm)  N-Pha(deg)  V-Amp(mm)  V-Pha(deg) \n'
    hf.write(cnv_str)
    hf.close()

    # Write Convolution Results to File
    if (loadfile_format == "bbox"):
        np.savetxt(cnv_body,all_cnv_data,fmt=["%s"] + ["%.8f",]*12,delimiter="      ")
    else:
        np.savetxt(cnv_body,all_cnv_data,fmt=["%s"] + ["%.8f",]*8,delimiter="      ")

    # Combine Header and Body Files
    filenames_cnv = [cnv_head, cnv_body]
    with open(cnv_file,'w') as outfile:
        for fname in filenames_cnv:
            with open(fname) as infile:
                outfile.write(infile.read())

    # Remove Header and Body Files
    os.remove(cnv_head)
    os.remove(cnv_body)

    # Return Amplitude and Phase Response Values
    return eamp,epha,namp,npha,vamp,vpha


