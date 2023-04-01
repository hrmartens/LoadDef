# *********************************************************************
# FUNCTION TO COMPUTE LOAD GREENS FUNCTIONS 
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
import math
import sys
import os
import datetime
from LOADGF.GF import harmonic_degree_summation
from LOADGF.utility import read_lln

"""
Parameters
----------
n : Spherical harmonic degrees

h : Vertical displacement load Love number

h_inf : First-order asymptotic value for the vertical-displacement load Love number

h_inf_p : Second-order asymptotic value for the vertical-displacement load Love number

nl : Horizontal displacement load Love number

l_inf : First-order asymptotic value for the horizontal-displacement load Love number

l_inf_p : Second-order asymptotic value for the horizontal-displacement load Love number

nk : Gravitational potential load Love number

k_inf : First-order asymptotic value for the potential load Love number

k_inf_p : Second-order asymptotic value for the potential load Love number

theta : Angular distances at which to comptue the load Green's functions (must be in list format)
    Default is Farrell spacing
        theta = [0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.16, 0.2, 0.25, 0.3, \
                0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, \
                10.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, \
                110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0]
 
a : Earth mean radius
    Default is 6371000.

me : Earth mean mass
    Default is 5.976E24

lmda_surface : Value of lambda elastic (Lame) parameter at Earth's surface
    Default is 3.422E10

mu_surface : Value of the shear modulus at Earth's surface
    Default is 2.662E10

g_surface : Value of acceleration due to gravity at Earth's surface
    Default is 9.81

disk_factor : Option to include a disk factor, to improve convergence of the LGFs in the far field 
    Default is True

angdist : Angular distance (in degrees) beyond which the disk factor will be included (if disk_factor = True)
    Default is 10.

disk_size : Radius of the disk in degrees
    Default is 0.1

apply_taper : Option to apply a taper to the series summation in the computation of the LGFs
              The taper mimics the recursive averaging desribed in Guo et al. 2004 [Eqs. 23 and 24]
    Default is True

loadmass : NEW option to provide the mass of a disk load. 
           Used ONLY when computing the analytical solution for a disk load. 
           Currently implemented only for displacement LGFs (vertical and horizontal displacement). 
    Default is None

grn_out : Extension for the output Green's function files 
    Default is 'grn.txt'
"""

def main(lln_file,rank,comm,size,grn_out='grn.txt',theta=None,a=6371000.,me=5.976E24,lmda_surface=3.422E10,mu_surface=2.662E10,g_surface=9.81,disk_factor=True,angdist=10.,disk_size=0.1,apply_taper=True,loadmass=None,max_theta=200.):

    # Generate Array of Angular Distances, if None Provided
    if theta is None:
        # Farrell theta spacing
        theta = [0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.16, 0.2, 0.25, 0.3, \
                0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, \
                10.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, \
                110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0]

    # Convert Theta List to Array
    theta = np.asarray(theta)
 
    # :: MPI ::
    # Determine the Chunk Sizes for GF
    total_theta = len(theta)
    nominal_gfload = total_theta // size
    if rank == size - 1:
        procNgf = total_theta - rank * nominal_gfload
    else:
        procNgf = nominal_gfload

    # If I'm the Master, I Initialize the GF Arrays
    if (rank == 0):

        # Print Status
        print(":: Computing Greens Functions. Please Wait...")
        print(" ")

        # Read in Data from Love Number File
        n,h,nl,nk,hp,hpp,lp,lpp,kp,kpp,a,me,lmda_surface,mu_surface,g_surface = read_lln.main(lln_file) 

        # Shuffle the Angular Distances to Improve Parallel Computing Efficiency 
        #  If Disk Factor is Applied, then Larger Angular Distances May Take Longer to Compute
        theta_mix = theta.copy()
        np.random.shuffle(theta_mix)

        # Initialize Arrays
        u              = np.empty(len(theta))
        v              = np.empty(len(theta))
        u_norm         = np.empty(len(theta))
        v_norm         = np.empty(len(theta))
        u_cm           = np.empty(len(theta))
        v_cm           = np.empty(len(theta))
        u_norm_cm      = np.empty(len(theta))
        v_norm_cm      = np.empty(len(theta))
        u_cf           = np.empty(len(theta))
        v_cf           = np.empty(len(theta))
        u_norm_cf      = np.empty(len(theta))
        v_norm_cf      = np.empty(len(theta))
        gE             = np.empty(len(theta))
        gE_norm        = np.empty(len(theta))
        gE_cm          = np.empty(len(theta))
        gE_cm_norm     = np.empty(len(theta))
        gE_cf          = np.empty(len(theta))
        gE_cf_norm     = np.empty(len(theta))
        tE             = np.empty(len(theta))
        tE_norm        = np.empty(len(theta))
        tE_cm          = np.empty(len(theta))
        tE_cm_norm     = np.empty(len(theta))
        tE_cf          = np.empty(len(theta))
        tE_cf_norm     = np.empty(len(theta))
        e_tt           = np.empty(len(theta))
        e_ll           = np.empty(len(theta))
        e_rr           = np.empty(len(theta))
        e_tt_cm        = np.empty(len(theta))
        e_ll_cm        = np.empty(len(theta))
        e_rr_cm        = np.empty(len(theta))
        e_tt_cf        = np.empty(len(theta))
        e_ll_cf        = np.empty(len(theta))
        e_rr_cf        = np.empty(len(theta))
        e_tt_norm      = np.empty(len(theta))
        e_ll_norm      = np.empty(len(theta))
        e_rr_norm      = np.empty(len(theta))
        e_tt_cm_norm   = np.empty(len(theta))
        e_ll_cm_norm   = np.empty(len(theta))
        e_rr_cm_norm   = np.empty(len(theta))
        e_tt_cf_norm   = np.empty(len(theta))
        e_ll_cf_norm   = np.empty(len(theta))
        e_rr_cf_norm   = np.empty(len(theta))
        g_N            = np.empty(len(theta))
        t_N            = np.empty(len(theta))

    # If I'm a Worker, I Know Nothing About the Data
    else: 
 
        u = v = u_norm = v_norm = u_cm = v_cm = u_norm_cm = v_norm_cm = u_cf = v_cf = u_norm_cf = v_norm_cf = theta_mix = None
        gE = gE_norm = gE_cm = gE_cm_norm = gE_cf = gE_cf_norm = tE = tE_norm = tE_cm = tE_cm_norm = tE_cf = tE_cf_norm = None
        e_tt = e_ll = e_rr = e_tt_cm = e_ll_cm = e_rr_cm = e_tt_cf = e_ll_cf = e_rr_cf = g_N = t_N = None
        e_tt_norm = e_ll_norm = e_rr_norm = e_tt_cm_norm = e_ll_cm_norm = e_rr_cm_norm = e_tt_cf_norm = e_ll_cf_norm = e_rr_cf_norm = None
        n = h = nl = nk = hp = hpp = lp = lpp = kp = kpp = a = me = lmda_surface = mu_surface = g_surface = None

    # Create a Data Type for Greens Functions
    gftype = MPI.DOUBLE.Create_contiguous(1)
    gftype.Commit()

    # Gather the Processor Workloads for All Processors
    sendcounts = comm.gather(procNgf, root=0)

    # Scatter the Theta Values
    t_sub = np.empty((procNgf,))
    comm.Scatterv([theta_mix, (sendcounts, None), gftype], t_sub, root=0)
    
    # All Processors Get Certain Arrays and Parameters; Broadcast Them
    n = comm.bcast(n, root=0)
    a = comm.bcast(a, root=0)
    me = comm.bcast(me, root=0)
    theta = comm.bcast(theta, root=0)
    lmda_surface = comm.bcast(lmda_surface, root=0)
    mu_surface = comm.bcast(mu_surface, root=0)
    g_surface = comm.bcast(g_surface, root=0)
    h = comm.bcast(h,root=0)
    nl = comm.bcast(nl,root=0)
    nk = comm.bcast(nk,root=0)
    hp = comm.bcast(hp,root=0)
    hpp = comm.bcast(hpp,root=0)
    lp = comm.bcast(lp,root=0)
    lpp = comm.bcast(lpp,root=0)
    kp = comm.bcast(kp,root=0)
    kpp = comm.bcast(kpp,root=0)

    # Initialize Arrays
    u_sub = np.empty((procNgf,))
    v_sub = np.empty((procNgf,))
    unorm_sub = np.empty((procNgf,))
    vnorm_sub = np.empty((procNgf,))
    u_cm_sub = np.empty((procNgf,))
    v_cm_sub = np.empty((procNgf,))
    unorm_cm_sub = np.empty((procNgf,))
    vnorm_cm_sub = np.empty((procNgf,))
    u_cf_sub = np.empty((procNgf,))
    v_cf_sub = np.empty((procNgf,))
    unorm_cf_sub = np.empty((procNgf,))
    vnorm_cf_sub = np.empty((procNgf,))
    gE_sub = np.empty((procNgf,))
    gE_norm_sub = np.empty((procNgf,))
    gE_cm_sub = np.empty((procNgf,))
    gE_cm_norm_sub = np.empty((procNgf,))
    gE_cf_sub = np.empty((procNgf,))
    gE_cf_norm_sub = np.empty((procNgf,))
    tE_sub = np.empty((procNgf,))
    tE_norm_sub = np.empty((procNgf,))
    tE_cm_sub = np.empty((procNgf,))
    tE_cm_norm_sub = np.empty((procNgf,))
    tE_cf_sub = np.empty((procNgf,))
    tE_cf_norm_sub = np.empty((procNgf,))
    ett_sub = np.empty((procNgf,))
    ell_sub = np.empty((procNgf,))
    err_sub = np.empty((procNgf,))
    ett_cm_sub = np.empty((procNgf,))
    ell_cm_sub = np.empty((procNgf,))
    err_cm_sub = np.empty((procNgf,))
    ett_cf_sub = np.empty((procNgf,))
    ell_cf_sub = np.empty((procNgf,))
    err_cf_sub = np.empty((procNgf,))
    ett_norm_sub = np.empty((procNgf,))
    ell_norm_sub = np.empty((procNgf,))
    err_norm_sub = np.empty((procNgf,))
    ett_cm_norm_sub = np.empty((procNgf,))
    ell_cm_norm_sub = np.empty((procNgf,))
    err_cm_norm_sub = np.empty((procNgf,))
    ett_cf_norm_sub = np.empty((procNgf,))
    ell_cf_norm_sub = np.empty((procNgf,))
    err_cf_norm_sub = np.empty((procNgf,))
    g_N_sub = np.empty((procNgf,))
    t_N_sub = np.empty((procNgf,))

    # Loop Through All Theta Values Assigned to this Rank
    for jj in range(0,procNgf):
        current_t = t_sub[jj]
        # CE Frame
        print('Working on Angular Distance: %9s | Reference Frame: CE | Number: %6d of %6d | Rank: %6d' %(str(current_t), (jj+1), len(t_sub), rank))
        u_sub[jj],v_sub[jj],unorm_sub[jj],vnorm_sub[jj],gE_sub[jj],gE_norm_sub[jj],tE_sub[jj],tE_norm_sub[jj],\
            ett_sub[jj],ell_sub[jj],err_sub[jj],ett_norm_sub[jj],ell_norm_sub[jj],err_norm_sub[jj],g_N_sub[jj],t_N_sub[jj] = \
            harmonic_degree_summation.main(n,a,me,current_t,h,hp,hpp,nl,lp,lpp,nk,kp,kpp,'CE',lmda_surface,mu_surface,g_surface,disk_factor,angdist,disk_size,apply_taper,max_theta=max_theta)
        # CM Frame
        print('Working on Angular Distance: %9s | Reference Frame: CM | Number: %6d of %6d | Rank: %6d' %(str(current_t), (jj+1), len(t_sub), rank))
        u_cm_sub[jj],v_cm_sub[jj],unorm_cm_sub[jj],vnorm_cm_sub[jj],gE_cm_sub[jj],gE_cm_norm_sub[jj],tE_cm_sub[jj],tE_cm_norm_sub[jj],\
            ett_cm_sub[jj],ell_cm_sub[jj],err_cm_sub[jj],ett_cm_norm_sub[jj],ell_cm_norm_sub[jj],err_cm_norm_sub[jj],g_N_sub[jj],t_N_sub[jj] = \
            harmonic_degree_summation.main(n,a,me,current_t,h,hp,hpp,nl,lp,lpp,nk,kp,kpp,'CM',lmda_surface,mu_surface,g_surface,disk_factor,angdist,disk_size,apply_taper,max_theta=max_theta)
        # CF Frame
        print('Working on Angular Distance: %9s | Reference Frame: CF | Number: %6d of %6d | Rank: %6d' %(str(current_t), (jj+1), len(t_sub), rank))
        u_cf_sub[jj],v_cf_sub[jj],unorm_cf_sub[jj],vnorm_cf_sub[jj],gE_cf_sub[jj],gE_cf_norm_sub[jj],tE_cf_sub[jj],tE_cf_norm_sub[jj],\
            ett_cf_sub[jj],ell_cf_sub[jj],err_cf_sub[jj],ett_cf_norm_sub[jj],ell_cf_norm_sub[jj],err_cf_norm_sub[jj],g_N_sub[jj],t_N_sub[jj] = \
            harmonic_degree_summation.main(n,a,me,current_t,h,hp,hpp,nl,lp,lpp,nk,kp,kpp,'CF',lmda_surface,mu_surface,g_surface,disk_factor,angdist,disk_size,apply_taper,max_theta=max_theta)

    # Gather Results
    comm.Gatherv(u_sub, [u, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(v_sub, [v, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(unorm_sub, [u_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(vnorm_sub, [v_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(u_cm_sub, [u_cm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(v_cm_sub, [v_cm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(unorm_cm_sub, [u_norm_cm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(vnorm_cm_sub, [v_norm_cm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(u_cf_sub, [u_cf, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(v_cf_sub, [v_cf, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(unorm_cf_sub, [u_norm_cf, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(vnorm_cf_sub, [v_norm_cf, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(gE_sub, [gE, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(gE_norm_sub, [gE_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(gE_cm_sub, [gE_cm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(gE_cm_norm_sub, [gE_cm_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(gE_cf_sub, [gE_cf, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(gE_cf_norm_sub, [gE_cf_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(tE_sub, [tE, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(tE_norm_sub, [tE_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(tE_cm_sub, [tE_cm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(tE_cm_norm_sub, [tE_cm_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(tE_cf_sub, [tE_cf, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(tE_cf_norm_sub, [tE_cf_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(ett_sub, [e_tt, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(ell_sub, [e_ll, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(err_sub, [e_rr, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(ett_cm_sub, [e_tt_cm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(ell_cm_sub, [e_ll_cm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(err_cm_sub, [e_rr_cm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(ett_cf_sub, [e_tt_cf, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(ell_cf_sub, [e_ll_cf, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(err_cf_sub, [e_rr_cf, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(ett_norm_sub, [e_tt_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(ell_norm_sub, [e_ll_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(err_norm_sub, [e_rr_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(ett_cm_norm_sub, [e_tt_cm_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(ell_cm_norm_sub, [e_ll_cm_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(err_cm_norm_sub, [e_rr_cm_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(ett_cf_norm_sub, [e_tt_cf_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(ell_cf_norm_sub, [e_ll_cf_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(err_cf_norm_sub, [e_rr_cf_norm, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(g_N_sub, [g_N, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(t_N_sub, [t_N, (sendcounts, None), MPI.DOUBLE], root=0)
 
    # Make sure everyone has reported back before moving on
    comm.Barrier()

    # Free Data Type
    gftype.Free()

    # Perform Linear Interpolation and Print Output Files
    if (rank == 0):

        # Re-Organize Angular Distances
        narr,nidx = np.unique(theta_mix,return_index=True)
        theta = theta_mix[nidx]
        u = u[nidx]; u_norm = u_norm[nidx]; v = v[nidx]; v_norm = v_norm[nidx]
        u_cm = u_cm[nidx]; u_norm_cm = u_norm_cm[nidx]; v_cm = v_cm[nidx]; v_norm_cm = v_norm_cm[nidx]
        u_cf = u_cf[nidx]; u_norm_cf = u_norm_cf[nidx]; v_cf = v_cf[nidx]; v_norm_cf = v_norm_cf[nidx]
        gE = gE[nidx]; gE_norm = gE_norm[nidx]
        gE_cm = gE_cm[nidx]; gE_cm_norm = gE_cm_norm[nidx] 
        gE_cf = gE_cf[nidx]; gE_cf_norm = gE_cf_norm[nidx]
        tE = tE[nidx]; tE_norm = tE_norm[nidx]
        tE_cm = tE_cm[nidx]; tE_cm_norm = tE_cm_norm[nidx]
        tE_cf = tE_cf[nidx]; tE_cf_norm = tE_cf_norm[nidx]
        e_tt = e_tt[nidx]; e_ll = e_ll[nidx]; e_rr = e_rr[nidx]
        e_tt_cm = e_tt_cm[nidx]; e_ll_cm = e_ll_cm[nidx]; e_rr_cm_ = e_rr_cm[nidx]
        e_tt_cf = e_tt_cf[nidx]; e_ll_cf = e_ll_cf[nidx]; e_rr_cf = e_rr_cf[nidx]
        e_tt_norm = e_tt_norm[nidx]; e_ll_norm = e_ll_norm[nidx]; e_rr_norm = e_rr_norm[nidx]
        e_tt_cm_norm = e_tt_cm_norm[nidx]; e_ll_cm_norm = e_ll_cm_norm[nidx]; e_rr_cm_norm = e_rr_cm_norm[nidx]
        e_tt_cf_norm = e_tt_cf_norm[nidx]; e_ll_cf_norm = e_ll_cf_norm[nidx]; e_rr_cf_norm = e_rr_cf_norm[nidx]
        g_N = g_N[nidx]; t_N = t_N[nidx]

        # Prepare Output Files
        grn_file_ce = ("../output/Greens_Functions/ce_" + grn_out)
        grn_file_cm = ("../output/Greens_Functions/cm_" + grn_out)
        grn_file_cf = ("../output/Greens_Functions/cf_" + grn_out)
        grn_head_ce = ("../output/Greens_Functions/"+str(np.random.randint(500))+"ce_head.txt")
        grn_head_cm = ("../output/Greens_Functions/"+str(np.random.randint(500))+"cm_head.txt")
        grn_head_cf = ("../output/Greens_Functions/"+str(np.random.randint(500))+"cf_head.txt")
        grn_body_ce = ("../output/Greens_Functions/"+str(np.random.randint(500))+"ce_body.txt")
        grn_body_cm = ("../output/Greens_Functions/"+str(np.random.randint(500))+"cm_body.txt")
        grn_body_cf = ("../output/Greens_Functions/"+str(np.random.randint(500))+"cf_body.txt")

        # For analytical disk, multiply by the mass of the load (for now, only for displacement)
        if loadmass is not None: 
            u *= loadmass
            v *= loadmass
            u_cm *= loadmass
            v_cm *= loadmass
            u_cf *= loadmass
            v_cf *= loadmass

        # Prepare Data for Output
        all_ce_data = np.column_stack((theta,u,v,u_norm,v_norm,gE_norm,tE_norm,e_rr_norm,e_tt_norm,e_ll_norm,g_N,t_N))
        all_cm_data = np.column_stack((theta,u_cm,v_cm,u_norm_cm,v_norm_cm,gE_cm_norm,tE_cm_norm,e_rr_cm_norm,e_tt_cm_norm,e_ll_cm_norm,g_N,t_N))
        all_cf_data = np.column_stack((theta,u_cf,v_cf,u_norm_cf,v_norm_cf,gE_cf_norm,tE_cf_norm,e_rr_cf_norm,e_tt_cf_norm,e_ll_cf_norm,g_N,t_N))

        # Write Header Info to File
        hf = open(grn_head_ce,'w')
        uh1 = ('------------------------------------------------------ \n')
        uh2 = (':: Load Greens Functions | CE Reference Frame \n')
        uh3 = (':: Computed at ' + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + '\n')
        uh4 = ('------------------------------------------------------ \n')
        hf.write(uh1); hf.write(uh2); hf.write(uh3); hf.write(uh4)
        grn_str_ce = 'theta | radial displacement (m/kg) | horizontal disp (m/kg) | r-disp*(a*th*10^12) | h-disp*(a*th*10^12) | gE*(10^18*(a*th)) | tE*(10^12*(a*th)^2) | e_rr*(10^18*(a*th)) | e_tt*(10^12*(a*th)^2) | e_ll*(10^12*(a*th)^2) | gN*(10^18*(a*th)) | tN*(10^12*(a*th)^2) \n'
        if loadmass is not None: 
            grn_str_ce = 'theta | radial displacement (m) | horizontal disp (m) | r-disp*(a*th*10^12) | h-disp*(a*th*10^12) | gE*(10^18*(a*th)) | tE*(10^12*(a*th)^2) | e_rr*(10^18*(a*th)) | e_tt*(10^12*(a*th)^2) | e_ll*(10^12*(a*th)^2) | gN*(10^18*(a*th)) | tN*(10^12*(a*th)^2) \n'
        hf.write(grn_str_ce)
        hf.close()
        hf = open(grn_head_cm,'w')
        uh1 = ('------------------------------------------------------ \n')
        uh2 = (':: Load Greens Functions | CM Reference Frame \n')
        uh3 = (':: Computed at ' + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + '\n')
        uh4 = ('------------------------------------------------------ \n')
        hf.write(uh1); hf.write(uh2); hf.write(uh3); hf.write(uh4)
        grn_str_cm = 'theta | radial displacement (m/kg) | horizontal disp (m/kg) | r-disp*(a*th*10^12) | h-disp*(a*th*10^12) | gE*(10^18*(a*th)) | tE*(10^12*(a*th)^2) | e_rr*(10^18*(a*th)) | e_tt*(10^12*(a*th)^2) | e_ll*(10^12*(a*th)^2) | gN*(10^18*(a*th)) | tN*(10^12*(a*th)^2) \n'
        if loadmass is not None:
            grn_str_cm = 'theta | radial displacement (m) | horizontal disp (m) | r-disp*(a*th*10^12) | h-disp*(a*th*10^12) | gE*(10^18*(a*th)) | tE*(10^12*(a*th)^2) | e_rr*(10^18*(a*th)) | e_tt*(10^12*(a*th)^2) | e_ll*(10^12*(a*th)^2) | gN*(10^18*(a*th)) | tN*(10^12*(a*th)^2) \n'
        hf.write(grn_str_cm)
        hf.close()
        hf = open(grn_head_cf,'w')
        uh1 = ('------------------------------------------------------ \n')
        uh2 = (':: Load Greens Functions | CF Reference Frame \n')
        uh3 = (':: Computed at ' + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + '\n')
        uh4 = ('------------------------------------------------------ \n')
        hf.write(uh1); hf.write(uh2); hf.write(uh3); hf.write(uh4)
        grn_str_cf = 'theta | radial displacement (m/kg) | horizontal disp (m/kg) | r-disp*(a*th*10^12) | h-disp*(a*th*10^12) | gE*(10^18*(a*th)) | tE*(10^12*(a*th)^2) | e_rr*(10^18*(a*th)) | e_tt*(10^12*(a*th)^2) | e_ll*(10^12*(a*th)^2) | gN*(10^18*(a*th)) | tN*(10^12*(a*th)^2) \n' 
        if loadmass is not None:
            grn_str_cf = 'theta | radial displacement (m) | horizontal disp (m) | r-disp*(a*th*10^12) | h-disp*(a*th*10^12) | gE*(10^18*(a*th)) | tE*(10^12*(a*th)^2) | e_rr*(10^18*(a*th)) | e_tt*(10^12*(a*th)^2) | e_ll*(10^12*(a*th)^2) | gN*(10^18*(a*th)) | tN*(10^12*(a*th)^2) \n'
        hf.write(grn_str_cf)
        hf.close()

        # Write CE Greens Functions to File
        np.savetxt(grn_body_ce, all_ce_data, fmt='%0.5f %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e',delimiter=" ")

        # Write CM Greens Functions to File
        np.savetxt(grn_body_cm, all_cm_data, fmt='%0.5f %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e',delimiter=" ")

        # Write CF Greens Functions to File
        np.savetxt(grn_body_cf, all_cf_data, fmt='%0.5f %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e',delimiter=" ")

        # Combine Header and Body Files
        filenames_ce = [grn_head_ce, grn_body_ce]
        with open(grn_file_ce,'w') as outfile:
            for fname in filenames_ce:
                with open(fname) as infile:
                    outfile.write(infile.read())
        filenames_cm = [grn_head_cm, grn_body_cm]
        with open(grn_file_cm,'w') as outfile:
            for fname in filenames_cm:
                with open(fname) as infile:
                    outfile.write(infile.read())
        filenames_cf = [grn_head_cf, grn_body_cf]
        with open(grn_file_cf,'w') as outfile:
            for fname in filenames_cf:
                with open(fname) as infile:
                    outfile.write(infile.read())

        # Remove Header and Body Files
        os.remove(grn_head_ce)
        os.remove(grn_body_ce)
        os.remove(grn_head_cm)
        os.remove(grn_body_cm)
        os.remove(grn_head_cf)
        os.remove(grn_body_cf)

        # Return Greens Functions
        return u,v,u_norm,v_norm,u_cm,v_cm,u_norm_cm,v_norm_cm,u_cf,v_cf,u_norm_cf,v_norm_cf,gE,gE_norm,gE_cm,gE_cm_norm,\
            gE_cf,gE_cf_norm,tE,tE_norm,tE_cm,tE_cm_norm,tE_cf,tE_cf_norm,\
            e_tt,e_ll,e_rr,e_tt_norm,e_ll_norm,e_rr_norm,e_tt_cm,e_ll_cm,e_rr_cm,e_tt_cm_norm,e_ll_cm_norm,e_rr_cm_norm,\
            e_tt_cf,e_ll_cf,e_rr_cf,e_tt_cf_norm,e_ll_cf_norm,e_rr_cf_norm,g_N,t_N

    else: 

        # If I'm a Worker, I Return Nothing
        return


