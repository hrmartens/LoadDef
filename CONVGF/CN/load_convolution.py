# *********************************************************************
# FUNCTION TO COMPUTE LOAD-INDUCED SURFACE DISPLACEMENTS
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
from mpi4py import MPI
import numpy as np
import datetime
import scipy as sc
import netCDF4
from scipy import interpolate
from CONVGF.utility import read_greens_fcn_file
from CONVGF.utility import read_greens_fcn_file_norm
from CONVGF.utility import normalize_greens_fcns
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
from CONVGF.utility import read_lsmask
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

def main(grn_file,norm_flag,load_files,regular,lslat,lslon,lsmask,rlat,rlon,stname,cnv_out,lsmask_type,loadfile_format,rank,procN,comm,\
    load_density=1030.0,rad=6371000.,delinc1=0.0002,delinc2=0.001,delinc3=0.01,delinc4=0.1,delinc5=0.5,delinc6=1.0,\
    izb=0.02,z2b=0.1,z3b=1.0,z4b=10.0,z5b=90.0,azminc=0.5,mass_cons=False):
 
    # Determine Number of Load Files
    if isinstance(load_files,float) == True:
        numel = 1
    else: 
        numel = len(load_files)

    # Only the Main Processor Will Do This:
    if (rank == 0):

        # Print Number of Files Read In
        display = ':: Number of Files Read In: ' + repr(numel)
        print(display)
        print(" ")

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
                slat,nlat,wlon,elon = np.loadtxt(loadgrid,usecols=(0,1,2,3),unpack=True)
            elif (lcext == 'nc'):
                f = netCDF4.Dataset(loadgrid)
                file_ids = f.variables['cell_ids'][:]
                slat = f.variables['slatitude'][:]
                nlat = f.variables['nlatitude'][:]
                wlon = f.variables['wlongitude'][:]
                elon = f.variables['elongitude'][:]
                f.close()
            # Ensure that Bounding Box Longitudes are in Range 0-360
            for yy in range(0,len(wlon)):
                if (wlon[yy] < 0.):
                    wlon[yy] += 360.
                if (elon[yy] < 0.):
                    elon[yy] += 360.
            # Generate an array of cell indices
            file_idx = np.linspace(0,len(file_ids),num=(len(file_ids)),endpoint=False)
            np.random.shuffle(file_idx)
        else:
            # Generate an Array of File Indices
            file_idx = np.linspace(0,numel,num=numel,endpoint=False)
            np.random.shuffle(file_idx)
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
        xr = np.linspace(0.0001,3.,1000)

        # Generate Integration Mesh
        print(':: Generating the Integration Mesh. Please Wait...')
        gldel,glazm,ldel,lazm,unit_area = generate_integration_mesh.main(delinc1,delinc2, \
            delinc3,delinc4,delinc5,delinc6,izb,z2b,z3b,z4b,z5b,azminc)

        # Integrate Greens Functions
        uint,vint = integrate_greens_fcns.main(gldel,glazm,ldel,lazm,tck_gfu,tck_gfv)

        # Compute Geographic Coordinates of Integration Mesh Cell Midpoints
        ilat,ilon,iarea = intmesh2geogcoords.main(rlat,rlon,ldel,lazm,unit_area)

        # Ensure that Station Location is in Range 0-360
        if (rlon < 0.):
            rlon += 360.

        # Determine the Land-Sea Mask: Interpolate onto Mesh
        print(':: Interpolating the Land-Sea Mask. Please Wait...')
        print(':: Number of Grid Points: %s | Size of LSMask: %s' %(str(len(ilat)), str(lsmask.shape)))
        lsmk = interpolate_lsmask.main(ilat,ilon,lslat,lslon,lsmask)
        print(':: Finished LSMask Interpolation.')

        # Compute Angular Distance and Azimuth at Receiver Due to Load
        delta,haz = compute_angularDist_azimuth.main(rlat,rlon,ilat,ilon)

        # Compute Greens Functions Specific to Receiver and Grid (Geographic Coordinates)
        ur,ue,un = compute_specific_greens_fcns.main(haz,uint,vint)

    # If I'm a Worker, I Know Nothing About the Data
    else:
 
        file_idx = file_ids = eamp = epha = namp = npha = vamp = vpha = None
        ldel = lazm = uint = vint = ilat = ilon = iarea = delta = haz = ur = ue = un = lsmk = None
        slat = nlat = wlon = elon = None

    # Create a Data Type for the Convolution Results
    cntype = MPI.DOUBLE.Create_contiguous(1)
    cntype.Commit()

    # Gather the Processor Workloads for All Processors
    sendcounts = comm.gather(procN, root=0)

    # Scatter the File Locations (By Index)
    d_sub = np.empty((procN,))
    comm.Scatterv([file_idx, (sendcounts, None), cntype], d_sub, root=0)

    # All Processors Get Certain Arrays and Parameters; Broadcast Them
    ilat  = comm.bcast(ilat, root=0)
    ilon  = comm.bcast(ilon, root=0)
    iarea = comm.bcast(iarea, root=0)
    ur    = comm.bcast(ur, root=0)
    ue    = comm.bcast(ue, root=0)
    un    = comm.bcast(un, root=0)
    load_density  = comm.bcast(load_density, root=0)
    lsmk  = comm.bcast(lsmk, root=0)
    file_ids = comm.bcast(file_ids, root=0)
    file_idx = comm.bcast(file_idx, root=0)
    if (loadfile_format == "bbox"):
        slat = comm.bcast(slat, root=0)
        nlat = comm.bcast(nlat, root=0)
        wlon = comm.bcast(wlon, root=0)
        elon = comm.bcast(elon, root=0)

    # Loop Through the Files 
    eamp_sub = np.empty((len(d_sub),))
    epha_sub = np.empty((len(d_sub),))
    namp_sub = np.empty((len(d_sub),))
    npha_sub = np.empty((len(d_sub),))
    vamp_sub = np.empty((len(d_sub),))
    vpha_sub = np.empty((len(d_sub),))
    for ii in range(0,len(d_sub)):
        current_d = int(d_sub[ii]) # Index
        if (loadfile_format == "bbox"):
            cslat = slat[current_d]
            cnlat = nlat[current_d]
            cwlon = wlon[current_d]
            celon = elon[current_d]
            mfile = [cslat,cnlat,cwlon,celon]
            file_id = file_ids[current_d] # File Extension
            print(':: Working on Cell: %s | Number: %6d of %6d | Rank: %6d' %(file_id, (ii+1), len(d_sub), rank))
        else: 
            mfile = load_files[current_d] # Vector-Format 
            file_id = file_ids[current_d] # File Extension
            print(':: Working on File: %s | Number: %6d of %6d | Rank: %6d' %(file_id, (ii+1), len(d_sub), rank))
            # Check if Loadfile Exists
            if not (os.path.isfile(mfile)): # file does not exist
                eamp_sub[ii] = np.nan
                epha_sub[ii] = np.nan
                namp_sub[ii] = np.nan
                npha_sub[ii] = np.nan
                vamp_sub[ii] = np.nan
                vpha_sub[ii] = np.nan
                continue
        # Compute Convolution for Current File
        eamp_sub[ii],epha_sub[ii],namp_sub[ii],npha_sub[ii],vamp_sub[ii],vpha_sub[ii] = perform_convolution.main(\
            mfile,ilat,ilon,iarea,load_density,ur,ue,un,lsmk,lsmask_type,file_id,regular,mass_cons,loadfile_format,stname)

    # Gather Results
    comm.Gatherv(eamp_sub, [eamp, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(epha_sub, [epha, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(namp_sub, [namp, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(npha_sub, [npha, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(vamp_sub, [vamp, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(vpha_sub, [vpha, (sendcounts, None), MPI.DOUBLE], root=0)

    # Make Sure Everyone Has Reported Back Before Moving On
    comm.Barrier()

    # Free Data Type
    cntype.Free()

    # Print Output to Files and Return Variables
    if (rank == 0):

        # Re-organize Files
        narr,nidx = np.unique(file_idx,return_index=True)
        eamp = eamp[nidx]; namp = namp[nidx]; vamp = vamp[nidx]
        epha = epha[nidx]; npha = npha[nidx]; vpha = vpha[nidx]

        # Prepare Output Files
        if (lsmask_type == 2):
            cnv_file = ("../output/Convolution/cn_LandOnly_" + cnv_out)
        elif (lsmask_type == 1):
            cnv_file = ("../output/Convolution/cn_OceanOnly_" + cnv_out)
        else:
            cnv_file = ("../output/Convolution/cn_LandAndOceans_" + cnv_out)
        cnv_head = ("../output/Convolution/"+str(np.random.randint(500))+"cn_head.txt")
        cnv_body = ("../output/Convolution/"+str(np.random.randint(500))+"cn_body.txt")
 
        # Prepare Data for Output (Create a Structured Array)
        rlat_arr = np.ones((len(eamp),)) * rlat
        rlon_arr = np.ones((len(eamp),)) * rlon
        if (loadfile_format == "bbox"):
            all_cnv_data = np.array(list(zip(file_ids,rlat_arr,rlon_arr,eamp,epha,namp,npha,vamp,vpha,slat,nlat,wlon,elon)), dtype=[('file_ids','U25'), \
                ('rlat_arr',float),('rlon_arr',float),('eamp',float),('epha',float),('namp',float),('npha',float), \
                ('vamp',float),('vpha',float),('slat',float),('nlat',float),('wlon',float),('elon',float)])
        else:
            all_cnv_data = np.array(list(zip(file_ids,rlat_arr,rlon_arr,eamp,epha,namp,npha,vamp,vpha)), dtype=[('file_ids','U25'), \
                ('rlat_arr',float),('rlon_arr',float),('eamp',float),('epha',float),('namp',float),('npha',float), \
                ('vamp',float),('vpha',float)])

        # Write Header Info to File
        hf = open(cnv_head,'w')
        if (loadfile_format == "bbox"):
            cnv_str = 'Extension/Epoch  Lat(+N,deg)  Lon(+E,deg)  E-Amp(mm)  E-Pha(deg)  N-Amp(mm)  N-Pha(deg)  V-Amp(mm)  V-Pha(deg)  S-Lat(deg)  N-Lat(deg)  W-Lon(deg)  E-Lon(deg) \n'
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

    else:

        # For Worker Ranks, Return Nothing
        return


