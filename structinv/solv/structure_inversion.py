# *********************************************************************
# FUNCTION TO INVERT OBSERVED DISPLACEMENTS FOR Earth Structure
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
from structinv.utility import read_structureDM
from structinv.solv import perform_inversion
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

uonly  :  Dataset contains only vertical components [NOTE: It is always assumed that the Design Matrix is built based on three components; this parameter applies only to the data!]
    Default is False (assumed that dataset contains three components -- e,n,u) 
 
imaginary  :  A flag to indicate whether (1) only real or (2) both real and imaginary components are included in the Design Matrix. 
    Default is False (assumed that the data are displacements at a moment in time, not harmonics)

outfile  :  Suffix for output file
    Default is ".txt"

"""

def main(dm_file,data_files,startmod,planet_model,rank,procN,comm,tikhonov='zeroth',alpha=1.,beta=1.,outfile=".txt",uonly=False,imaginary=False,pme=False):

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
        design_matrix,sta_comp_ids,sta_comp_lat,sta_comp_lon,pert_rad_bot,pert_rad_top,pert_param = read_structureDM.main(dm_file)

        # Remove Spatial Component from Station Names
        sta_ids = []
        for bb in range(0,len(sta_comp_ids)):
            cid = sta_comp_ids[bb]
            if (imaginary == False): 
                sta_ids.append(cid[0:-1])
            else: 
                sta_ids.append(cid[0:-3])
        sta_ids = np.asarray(sta_ids)

        # Initialize Array
        model_vector = np.empty((len(file_ids),len(pert_param)))

    # If I'm a Worker, I Know Nothing About the Data
    else:
  
        design_matrix = sta_ids = sta_comp_ids = sta_comp_lat = sta_comp_lon = pert_rad_bot = pert_rad_top = pert_param = None
        file_idx = file_ids = model_vector = None

    # Create a Data Type for the Datafiles
    dftype = MPI.DOUBLE.Create_contiguous(1)
    dftype.Commit()

    # Create a Data Type for the Model Vector
    mv_size = len(pert_param)
    mvtype = MPI.DOUBLE.Create_contiguous(mv_size)
    mvtype.Commit()

    # Gather the Processor Workloads for All Processors
    sendcounts = comm.gather(procN, root=0)

    # Scatter the File Locations (By Index)
    d_sub = np.empty((procN,))
    comm.Scatterv([file_idx, (sendcounts, None), dftype], d_sub, root=0)

    # All Processors Get Certain Arrays and Parameters; Broadcast Them
    design_matrix = comm.bcast(design_matrix, root=0)
    sta_ids       = comm.bcast(sta_ids, root=0)
    sta_comp_ids  = comm.bcast(sta_comp_ids, root=0)
    sta_comp_lat  = comm.bcast(sta_comp_lat, root=0)
    sta_comp_lon  = comm.bcast(sta_comp_lon, root=0)
    pert_rad_bot  = comm.bcast(pert_rad_bot, root=0)
    pert_rad_top  = comm.bcast(pert_rad_top, root=0)
    pert_param    = comm.bcast(pert_param, root=0)
    file_ids      = comm.bcast(file_ids, root=0)
    file_idx      = comm.bcast(file_idx, root=0)

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
        mv_sub[ii,:] = perform_inversion.main(mfile,file_id,startmod,design_matrix,sta_ids,sta_comp_ids,sta_comp_lat,sta_comp_lon,pert_rad_bot,pert_rad_top,pert_param,tikhonov,alpha,beta,uonly=uonly,inc_imag=imaginary,pme=pme)

    # Gather Results
    comm.Gatherv(mv_sub, [model_vector, (sendcounts, None), mvtype], root=0)

    # Make Sure Everyone Has Reported Back Before Moving On
    comm.Barrier()

    # Free Data Type
    mvtype.Free()

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

            # Current model vector (perturbation to the original model)
            cmv = model_vector[kk,:]
         
            # Prepare Output Files
            inv_out = (cfileid + outfile)
            inv_file = ("../output/Inversion/Structure/iv_" + inv_out)
            inv_head = ("../output/Inversion/Structure/"+str(np.random.randint(500))+"cn_head.txt")
            inv_body = ("../output/Inversion/Structure/"+str(np.random.randint(500))+"cn_body.txt")
 
            # Prepare Data for Output (Create a Structured Array)
            all_inv_data = np.array(list(zip(pert_param,pert_rad_bot,pert_rad_top,cmv)), dtype=[('pert_param','U25'), \
                ('pert_rad_bot',float),('pert_rad_top',float),('cmv',float)])
 
            # Write Header Info to File
            hf = open(inv_head,'w')
            inv_str = 'MaterialParameter  LayerBottomRadius(km)  LayerTopRadius(km)  Perturbation  \n'
            hf.write(inv_str)
            hf.close()

            # Write Convolution Results to File
            np.savetxt(inv_body,all_inv_data,fmt=["%s"] + ["%.8f",]*3,delimiter="      ")
 
            # Combine Header and Body Files
            filenames_inv = [inv_head, inv_body]
            with open(inv_file,'w') as outfile:
                for fname in filenames_inv:
                    with open(fname) as infile:
                        outfile.write(infile.read())
 
            # Remove Header and Body Files
            os.remove(inv_head)
            os.remove(inv_body)

        # Return Model Vector
        return model_vector

    else:

        # For Worker Ranks, Return Nothing
        return


