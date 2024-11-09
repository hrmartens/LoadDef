# *********************************************************************
# FUNCTION TO INVERT OBSERVED DISPLACEMENTS FOR SURFACE LOAD
# 
# Copyright (c) 2021-2024: HILARY R. MARTENS
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
import scipy as sc
import netCDF4
from loadinv.utility import read_loadDesignMatrix
from loadinv.solv import perform_inversion
import sys
import os

"""
Estimate surface-load distribution from observed surface displacements.

Parameters
----------

alpha  :  Coefficient for Second-Order Tikhonov Regularization
    Default is 1

beta   :  Coefficient for Zeroth-Order Tikhonov Regularization
    Default is 1

tikhonov  :  Order of Tikhonov Regularization to Apply (Options: 'zeroth', 'second', 'zeroth_second', 'none')
    Default is zeroth

reference_height  :  Reference Load Height used to Generate the Design Matrix (units: m) [see run_dm_load.py in LoadDef]
    Default is 1

reference_density  :  Reference Load Density used to Generate the Design Matrix (units: kg/m^3) [see run_dm_load.py in LoadDef]
    Default is 1000

uonly  :  Dataset contains only vertical components [NOTE: It is always assumed that the Design Matrix is built based on three components; this parameter applies only to the data!]
    Defautl is False (assumed that dataset contains three components -- e,n,u)

outfile  :  Suffix for output file
    Default is ".txt"

"""

def main(dm_file,data_files,rank,procN,comm,reference_height=1.,reference_density=1000.,tikhonov='zeroth',alpha=1.,beta=1.,outfile=".txt",uonly=False):

    # Determine Number of Datafiles
    if isinstance(data_files,float) == True:
        numel = 1
    else: 
        numel = len(data_files)

    # Only the Main Processor Will Do This:
    if (rank == 0):

        # Print Number of Datafiles Read In
        display = ':: Number of Datafiles Read In: ' + repr(numel)
        print(display)
        print(" ")

        # Generate an Array of File Indices
        file_idx = np.linspace(0,numel,num=numel,endpoint=False)
        np.random.shuffle(file_idx)
        # Create Array of Filename Extensions
        file_ids = []
        for qq in range(0,numel):
            mfile = data_files[qq]
            str_components = mfile.split('/')
            cext = str_components[-1]
            dfext = mfile[-3::]
            if (dfext == "txt"):
                file_ids.append(cext[0:-4]) 
            else:
                print(':: Error. Invalid file format for datafiles. [load_inversion.py]')
                sys.exit()

        # Read in Design Matrix File
        design_matrix,sta_comp_ids,sta_comp_lat,sta_comp_lon,load_cell_ids,load_cell_lat,load_cell_lon = read_loadDesignMatrix.main(dm_file)

        # Remove Spatial Component from Station Names
        sta_ids = []
        for bb in range(0,len(sta_comp_ids)):
            cid = sta_comp_ids[bb]
            sta_ids.append(cid[0:-1])
        sta_ids = np.asarray(sta_ids)

        # Initialize Array
        model_vector = np.empty((len(file_ids),len(load_cell_ids)))

    # If I'm a Worker, I Know Nothing About the Data
    else:
 
        design_matrix = sta_ids = sta_comp_ids = sta_comp_lat = sta_comp_lon = load_cell_ids = load_cell_lat = load_cell_lon = None
        file_idx = file_ids = model_vector = None

    # All Processors Get Certain Arrays and Parameters; Broadcast Them
    design_matrix  = comm.bcast(design_matrix, root=0)
    sta_ids       = comm.bcast(sta_ids, root=0)
    sta_comp_ids  = comm.bcast(sta_comp_ids, root=0)
    sta_comp_lat  = comm.bcast(sta_comp_lat, root=0)
    sta_comp_lon  = comm.bcast(sta_comp_lon, root=0)
    load_cell_ids = comm.bcast(load_cell_ids, root=0)
    load_cell_lat = comm.bcast(load_cell_lat, root=0)
    load_cell_lon = comm.bcast(load_cell_lon, root=0)
    file_ids      = comm.bcast(file_ids, root=0)
    file_idx      = comm.bcast(file_idx, root=0)

    # Create a Data Type for the Datafiles
    dftype = MPI.DOUBLE.Create_contiguous(1)
    dftype.Commit()

    # Create a Data Type for the Model Vector
    mv_size = len(load_cell_ids)
    mvtype = MPI.DOUBLE.Create_contiguous(mv_size)
    mvtype.Commit()

    # Gather the Processor Workloads for All Processors
    sendcounts = comm.gather(procN, root=0)

    # Scatter the File Locations (By Index)
    d_sub = np.empty((procN,))
    comm.Scatterv([file_idx, (sendcounts, None), dftype], d_sub, root=0)

    # Loop Through the Files 
    mv_sub = np.empty((len(d_sub),mv_size))
    for ii in range(0,len(d_sub)):
        current_d = int(d_sub[ii]) # Index
        mfile = data_files[current_d] # Vector-Format 
        file_id = file_ids[current_d] # File Extension
        print(':: Working on File: %s | Number: %6d of %6d | Rank: %6d' %(file_id, (ii+1), len(d_sub), rank))
        # Check if Loadfile Exists
        if not (os.path.isfile(mfile)): # file does not exist
            mv_sub[ii,:] = np.nan
            continue
        # Perform Inversion for Current File
        mv_sub[ii,:] = perform_inversion.main(mfile,file_id,design_matrix,sta_ids,sta_comp_ids,sta_comp_lat,sta_comp_lon,load_cell_ids,load_cell_lat,load_cell_lon,tikhonov,alpha,beta,uonly=uonly)

    # Gather Results
    comm.Gatherv(mv_sub, [model_vector, (sendcounts, None), mvtype], root=0)

    # Make Sure Everyone Has Reported Back Before Moving On
    comm.Barrier()

    # Free Data Type
    mvtype.Free()
    dftype.Free()

    # Print Output to Files and Return Variables
    if (rank == 0):

        # Re-organize Files
        narr,nidx = np.unique(file_idx,return_index=True)
        model_vector = model_vector[nidx,:]

        # Loop through datafiles
        for kk in range(0,len(data_files)):

            # Current file
            cfile = data_files[kk]
            cfileid = file_ids[kk]

            # Current model vector (scaling factors for actual load)
            cmv = model_vector[kk,:]
         
            # Compute load pressure in 1000*N/m^2 (kPa)
            cmv_p = cmv * reference_height * reference_density * 9.81 / 1000.

            # Compute load heigh in m
            cmv_h = cmv * reference_height

            # Prepare Output Files
            inv_out = (cfileid + outfile)
            inv_file = ("../output/Inversion/SurfaceLoad/iv_" + inv_out)
            inv_head = ("../output/Inversion/SurfaceLoad/"+str(np.random.randint(500))+"cn_head.txt")
            inv_body = ("../output/Inversion/SurfaceLoad/"+str(np.random.randint(500))+"cn_body.txt")
 
            # Prepare Data for Output (Create a Structured Array)
            #all_inv_data = np.array(list(zip(load_cell_ids,load_cell_lat,load_cell_lon,cmv_h,cmv_p)), dtype=[('load_cell_ids','U25'), \
            #    ('load_cell_lat',float),('load_cell_lon',float),('cmv_h',float),('cmv_p',float)])
            all_inv_data = np.array(list(zip(load_cell_lat,load_cell_lon,cmv_h,cmv_p)), dtype=[('load_cell_lat',float),('load_cell_lon',float),('cmv_h',float),('cmv_p',float)])
  
            # Write Header Info to File
            hf = open(inv_head,'w')
            #inv_str = 'LoadCellID  LoadCellLat(+N,deg)  LoadCellLon(+E,deg)  LoadCellHeight(m)  LoadCellPressure(kPa)  \n'
            inv_str = 'LoadCellLat(+N,deg)  LoadCellLon(+E,deg)  LoadCellHeight(m)  LoadCellPressure(kPa)  \n'
            hf.write(inv_str)
            hf.close()

            # Write Convolution Results to File
            #np.savetxt(inv_body,all_inv_data,fmt=["%s"] + ["%.8f",]*4,delimiter="      ")
            np.savetxt(inv_body,all_inv_data,fmt=["%.8f",]*4,delimiter="      ")
 
            # Combine Header and Body Files
            filenames_inv = [inv_head, inv_body]
            with open(inv_file,'w') as outf:
                for fname in filenames_inv:
                    with open(fname) as infile:
                        outf.write(infile.read())
 
            # Remove Header and Body Files
            os.remove(inv_head)
            os.remove(inv_body)

        # Return Model Vector
        return model_vector

    else:

        # For Worker Ranks, Return Nothing
        return


