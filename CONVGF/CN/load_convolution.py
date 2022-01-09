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
    Default is 0.05

z3b     : zone 3 boundary (degrees; z2b<z3b<z4b)
    Default is 1.0

z4b     : zone 4 boundary (degrees; z3b<z4b<z5b)
    Default is 10.0

z5b     : zone 5 boundary (degrees; z4b<z5b<180)
    Default is 90.0

azminc  : azimuthal increment
    Default is 0.1

mass_cons  :  option to enforce mass conservation by removing the spatial mean from the load grid
    Default is False

"""

def main(grn_file,norm_flag,load_files,regular,rlat,rlon,stname,cnv_out,lsmask_file,lsmask_type,loadfile_format,rank,procN,comm,\
    load_density=1030.0,rad=6371000.,delinc1=0.0002,delinc2=0.001,delinc3=0.01,delinc4=0.1,delinc5=0.5,delinc6=1.0,\
    izb=0.02,z2b=0.05,z3b=1.0,z4b=10.0,z5b=90.0,azminc=0.1,mass_cons=False):

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

        # Generate an Array of File Indices
        file_idx = np.linspace(0,numel,num=numel,endpoint=False)
        np.random.shuffle(file_idx)

        # Create Array of Filename Extensions
        file_exts = []
        for qq in range(0,numel):
            mfile = load_files[qq]
            str_components = mfile.split('_')
            cext = str_components[-1]
            if (loadfile_format == "txt"):
                file_exts.append(cext[0:-4]) 
            elif (loadfile_format == "nc"):
                file_exts.append(cext[0:-3]) 
            else:
                print(':: Error. Invalid file format for load models. [load_convolution.py]')
                sys.exit()

        # Initialize Arrays
        eamp = np.empty((len(load_files),))
        epha = np.empty((len(load_files),))
        namp = np.empty((len(load_files),))
        npha = np.empty((len(load_files),))
        vamp = np.empty((len(load_files),))
        vpha = np.empty((len(load_files),))

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

        # Read in the Land-Sea Mask
        if (lsmask_type > 0):
            lslat,lslon,lsmask = read_lsmask.main(lsmask_file)
        else: 
            #lslat = ilat[0::10]
            #lslon = ilon[0::10]
            #lsmask = np.ones((len(lslat),)) * -1.
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
        lslon[neglon_idx] = lslon[neglon_idx] + 360.
 
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
 
        file_idx = file_exts = eamp = epha = namp = npha = vamp = vpha = None
        ldel = lazm = uint = vint = ilat = ilon = iarea = delta = haz = ur = ue = un = lsmk = None

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
    file_exts = comm.bcast(file_exts, root=0)
    file_idx = comm.bcast(file_idx, root=0)

    # Loop Through the Files 
    eamp_sub = np.empty((len(d_sub),))
    epha_sub = np.empty((len(d_sub),))
    namp_sub = np.empty((len(d_sub),))
    npha_sub = np.empty((len(d_sub),))
    vamp_sub = np.empty((len(d_sub),))
    vpha_sub = np.empty((len(d_sub),))
    for ii in range(0,len(d_sub)):
        current_d = int(d_sub[ii]) # Index
        mfile = load_files[current_d] # Vector-Format 
        file_ext = file_exts[current_d] # File Extension
        print('Working on File: %s | Number: %6d of %6d | Rank: %6d' %(file_ext, (ii+1), len(d_sub), rank))
        # Check if Loadfile Exists
        if (os.path.isfile(mfile)):
            # Compute Convolution for Current File
            eamp_sub[ii],epha_sub[ii],namp_sub[ii],npha_sub[ii],vamp_sub[ii],vpha_sub[ii] = perform_convolution.main(\
                mfile,ilat,ilon,iarea,load_density,ur,ue,un,lsmk,lsmask_type,file_ext,regular,mass_cons,loadfile_format)
        else: # File Does Not Exist
            eamp_sub[ii] = np.nan
            epha_sub[ii] = np.nan
            namp_sub[ii] = np.nan
            npha_sub[ii] = np.nan
            vamp_sub[ii] = np.nan
            vpha_sub[ii] = np.nan

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
        all_cnv_data = np.array(list(zip(file_exts,rlat_arr,rlon_arr,eamp,epha,namp,npha,vamp,vpha)), dtype=[('file_exts','U25'), \
            ('rlat_arr',float),('rlon_arr',float),('eamp',float),('epha',float),('namp',float),('npha',float), \
            ('vamp',float),('vpha',float)])

        # Write Header Info to File
        hf = open(cnv_head,'w')
        cnv_str = 'Extension/Epoch  Lat(+N,deg)  Lon(+E,deg)  E-Amp(mm)  E-Pha(deg)  N-Amp(mm)  N-Pha(deg)  V-Amp(mm)  V-Pha(deg) \n'
        hf.write(cnv_str)
        hf.close()

        # Write Convolution Results to File
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


