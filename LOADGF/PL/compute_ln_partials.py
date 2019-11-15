# *********************************************************************
# FUNCTION TO COMPUTE THE PARTIAL DERIVATIVES OF LOVE NUMBERS
# See: Okubo & Saito (1983); Martens et al. (2016, JGR-Solid Earth)
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
import math
import os
import sys
import datetime

# Import LoadDef Modules
from LOADGF.PL import ln_partials
from LOADGF.LN import prepare_planet_model
from scipy import interpolate
from scipy.integrate import simps
 
"""
Function to compute partial derivatives of Love numbers. 
For more information on the theory, see Okubo & Saito (1983) and Martens et al. (2016)

G : Universal Gravitational Constant
    Default is 6.672E-11 m^3/(kg*s^2)

period_hours : Tidal forcing period (in hours)
    Default is 12.42 (M2 period)

r_min : Minimum radius for variable Earth structural properties (meters)
    Default is 1000

kx : Order of the spline fit for the Earth model (1=linear; 3=cubic)
    Default is 1 (recommended)

delim : Delimiter for the Earth model file
    Default is None (Whitespace)

plot_figure : Reproduce Figure 1 from Okubo & Endo (1986) as a sanity check
    Default is False

par_out : Extension for the output filenames.
    Default is "partials.txt"
"""

def main(myn,sint,Yload,Ypot,Yshr,Ystr,hload,nlload,nkload,hpot,nlpot,nkpot,hshr,nlshr,nkshr,\
    hstr,nlstr,nkstr,emod_file,rank,comm,size,par_out='partials.txt',G=6.627E-11,r_min=1000.,kx=1,period_hours=12.42,delim=None,plot_figure=False):

    # Determine the Chunk Sizes for LLN
    total_lln = len(myn)
    nominal_load = total_lln // size # Floor Divide
    # Final Chunk Might Have Fewer or More Jobs
    if rank == size - 1:
        procN = total_lln - rank * nominal_load
    else: # Otherwise, Chunks are Size of nominal_load
        procN = nominal_load
 
    # Length of Radii Vector
    if (sint.size == sint.shape[0]):
        sys.exit("Error: Must have more than one radial distance.")
    else:  
        sint_length = sint.shape[1]

    if (rank == 0):

        # Print Update to Screen
        print(":: Computing Partial Derivatives. Please Wait..."); print("")

        # For SNREI Earth, Angular Frequency (omega) is Zero 
        # Azimuthal order is only utilized for a rotating Earth
        # The variables for each are included here as "place-holders" for future versions
        omega = 0
        order = 2

        # Prepare the Earth Model (read in, non-dimensionalize elastic parameters, etc.)
        r,mu,K,lmda,rho,g,tck_lnd,tck_mnd,tck_rnd,tck_gnd,s,lnd,mnd,rnd,gnd,s_min,small,\
            planet_radius,planet_mass,sic,soc,adim,gsdim,pi,piG,L_sc,R_sc,T_sc = \
            prepare_planet_model.main(emod_file,G,r_min,kx,file_delim=delim)

        # Define Forcing Period
        w = (1./(period_hours*3600.))*(2.*pi) # convert period to frequency (rad/sec)
        wnd = w*T_sc                         # non-dimensionalize
        ond = omega*T_sc

        # Initialize Arrays of Partial Derivatives
        dht_dmu  = np.empty((len(myn),sint_length))
        dlt_dmu  = np.empty((len(myn),sint_length))
        dkt_dmu  = np.empty((len(myn),sint_length))
        dht_dK   = np.empty((len(myn),sint_length))
        dlt_dK   = np.empty((len(myn),sint_length))
        dkt_dK   = np.empty((len(myn),sint_length))
        dht_drho = np.empty((len(myn),sint_length))
        dlt_drho = np.empty((len(myn),sint_length))
        dkt_drho = np.empty((len(myn),sint_length))
        dhl_dmu  = np.empty((len(myn),sint_length))
        dll_dmu  = np.empty((len(myn),sint_length))
        dkl_dmu  = np.empty((len(myn),sint_length))
        dhl_dK   = np.empty((len(myn),sint_length))
        dll_dK   = np.empty((len(myn),sint_length))
        dkl_dK   = np.empty((len(myn),sint_length))
        dhl_drho = np.empty((len(myn),sint_length))
        dll_drho = np.empty((len(myn),sint_length))
        dkl_drho = np.empty((len(myn),sint_length))
        dhs_dmu  = np.empty((len(myn),sint_length))
        dls_dmu  = np.empty((len(myn),sint_length))
        dks_dmu  = np.empty((len(myn),sint_length))
        dhs_dK   = np.empty((len(myn),sint_length))
        dls_dK   = np.empty((len(myn),sint_length))
        dks_dK   = np.empty((len(myn),sint_length))
        dhs_drho = np.empty((len(myn),sint_length))
        dls_drho = np.empty((len(myn),sint_length))
        dks_drho = np.empty((len(myn),sint_length))

    else: 

        # Workers Know Nothing About the Data
        dht_dmu = dlt_dmu = dkt_dmu = dht_dK = dlt_dK = dkt_dK = dht_drho = dlt_drho = dkt_drho = None
        dhl_dmu = dll_dmu = dkl_dmu = dhl_dK = dll_dK = dkl_dK = dhl_drho = dll_drho = dkl_drho = None
        dhs_dmu = dls_dmu = dks_dmu = dhs_dK = dls_dK = dks_dK = dhs_drho = dls_drho = dks_drho = None
        tck_lnd = tck_mnd = tck_rnd = tck_gnd = wnd = ond = piG = order = R_sc = T_sc = L_sc = None

    # Create a Data Type for the Harmonic Degrees
    lntype = MPI.DOUBLE.Create_contiguous(1)
    lntype.Commit() 

    # Create a Data Type for the Partials
    ptype = MPI.DOUBLE.Create_contiguous(sint_length)
    ptype.Commit()

    # Gather the Processor Workloads for All Processors
    sendcounts = comm.gather(procN, root=0)

    # Scatter the Harmonic Degrees
    n_sub = np.empty((procN,))
    comm.Scatterv([myn, (sendcounts, None), lntype], n_sub, root=0)

    # All Processors Get Certain Arrays and Parameters; Broadcast Them
    tck_lnd = comm.bcast(tck_lnd, root=0)
    tck_mnd = comm.bcast(tck_mnd, root=0)
    tck_rnd = comm.bcast(tck_rnd, root=0)
    tck_gnd = comm.bcast(tck_gnd, root=0)
    wnd = comm.bcast(wnd, root=0)
    ond = comm.bcast(ond, root=0)
    piG = comm.bcast(piG, root=0)
    order = comm.bcast(order, root=0)
    L_sc = comm.bcast(L_sc, root=0)
    T_sc = comm.bcast(T_sc, root=0)
    R_sc = comm.bcast(R_sc, root=0)

    # Loop Through Spherical Harmonic Degrees
    dht_dmu_sub  = np.empty((len(n_sub),sint_length))
    dlt_dmu_sub  = np.empty((len(n_sub),sint_length))
    dkt_dmu_sub  = np.empty((len(n_sub),sint_length))
    dht_dK_sub   = np.empty((len(n_sub),sint_length)) 
    dlt_dK_sub   = np.empty((len(n_sub),sint_length))
    dkt_dK_sub   = np.empty((len(n_sub),sint_length))
    dht_drho_sub = np.empty((len(n_sub),sint_length))
    dlt_drho_sub = np.empty((len(n_sub),sint_length))
    dkt_drho_sub = np.empty((len(n_sub),sint_length))
    dhl_dmu_sub  = np.empty((len(n_sub),sint_length))
    dll_dmu_sub  = np.empty((len(n_sub),sint_length))
    dkl_dmu_sub  = np.empty((len(n_sub),sint_length))
    dhl_dK_sub   = np.empty((len(n_sub),sint_length))
    dll_dK_sub   = np.empty((len(n_sub),sint_length))
    dkl_dK_sub   = np.empty((len(n_sub),sint_length))
    dhl_drho_sub = np.empty((len(n_sub),sint_length))
    dll_drho_sub = np.empty((len(n_sub),sint_length))
    dkl_drho_sub = np.empty((len(n_sub),sint_length))
    dhs_dmu_sub  = np.empty((len(n_sub),sint_length))
    dls_dmu_sub  = np.empty((len(n_sub),sint_length))
    dks_dmu_sub  = np.empty((len(n_sub),sint_length))
    dhs_dK_sub   = np.empty((len(n_sub),sint_length))
    dls_dK_sub   = np.empty((len(n_sub),sint_length))
    dks_dK_sub   = np.empty((len(n_sub),sint_length))
    dhs_drho_sub = np.empty((len(n_sub),sint_length))
    dls_drho_sub = np.empty((len(n_sub),sint_length))
    dks_drho_sub = np.empty((len(n_sub),sint_length))
    for ii in range(0,len(n_sub)):
        current_n = n_sub[ii]
        print('Partial Derivatives | Working on Harmonic Degree: %7s | Number: %6d of %6d | Rank: %6d' %(str(int(current_n)), (ii+1), len(n_sub), rank))
        nidx = int(current_n - min(myn)) # Determine Index (If the Range of 'n' Does Not Start at Zero)
        # Compute Integration Results for the Current Spherical Harmonic Degree, n
        dht_dmu_sub[ii,:],dlt_dmu_sub[ii,:],dkt_dmu_sub[ii,:],dht_dK_sub[ii,:],dlt_dK_sub[ii,:],dkt_dK_sub[ii,:],dht_drho_sub[ii,:],dlt_drho_sub[ii,:],dkt_drho_sub[ii,:],\
            dhl_dmu_sub[ii,:],dll_dmu_sub[ii,:],dkl_dmu_sub[ii,:],dhl_dK_sub[ii,:],dll_dK_sub[ii,:],dkl_dK_sub[ii,:],dhl_drho_sub[ii,:],dll_drho_sub[ii,:],dkl_drho_sub[ii,:],\
            dhs_dmu_sub[ii,:],dls_dmu_sub[ii,:],dks_dmu_sub[ii,:],dhs_dK_sub[ii,:],dls_dK_sub[ii,:],dks_dK_sub[ii,:],dhs_drho_sub[ii,:],dls_drho_sub[ii,:],dks_drho_sub[ii,:] = \
                ln_partials.main(current_n,sint[nidx,:],Yload[nidx,:,:],Ypot[nidx,:,:],Yshr[nidx,:,:],Ystr[nidx,:,:],hload[nidx],nlload[nidx],nkload[nidx],\
                hpot[nidx],nlpot[nidx],nkpot[nidx],hshr[nidx],nlshr[nidx],nkshr[nidx],hstr[nidx],nlstr[nidx],nkstr[nidx],\
                tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,order,plot_figure)

    # Gather Results    
    comm.Gatherv(dht_dmu_sub, [dht_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dlt_dmu_sub, [dlt_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dkt_dmu_sub, [dkt_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dht_dK_sub, [dht_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dlt_dK_sub, [dlt_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dkt_dK_sub, [dkt_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dht_drho_sub, [dht_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dlt_drho_sub, [dlt_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dkt_drho_sub, [dkt_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dhl_dmu_sub, [dhl_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dll_dmu_sub, [dll_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dkl_dmu_sub, [dkl_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dhl_dK_sub, [dhl_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dll_dK_sub, [dll_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dkl_dK_sub, [dkl_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dhl_drho_sub, [dhl_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dll_drho_sub, [dll_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dkl_drho_sub, [dkl_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dhs_dmu_sub, [dhs_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dls_dmu_sub, [dls_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dks_dmu_sub, [dks_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dhs_dK_sub, [dhs_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dls_dK_sub, [dls_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dks_dK_sub, [dks_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dhs_drho_sub, [dhs_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dls_drho_sub, [dls_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(dks_drho_sub, [dks_drho, (sendcounts, None), ptype], root=0)

    # Make Sure Everyone Has Reported Back Before Moving On
    comm.Barrier()

    # Free Data Types
    lntype.Free()
    ptype.Free()

    # Return Results and Print to File 
    if (rank == 0):
 
        # Loop through all Spherical Harmonic Degrees
        for jj in range(0,len(myn)):

            # Current Harmonic Degree
            cn = myn[jj]

            # Interpolate Parameters to 'sint' Locations
            imnd = interpolate.splev(sint[jj,:],tck_mnd)
            irnd = interpolate.splev(sint[jj,:],tck_rnd)
            ilnd = interpolate.splev(sint[jj,:],tck_lnd)
            iknd = ilnd + (2./3.)*imnd

            # Re-Dimensionalize
            ikappa = iknd*(R_sc*(L_sc**2)*(T_sc**(-2)))
            imu    = imnd*(R_sc*(L_sc**2)*(T_sc**(-2)))
            irho   = irnd*R_sc
            rint   = sint[jj,:]*L_sc/1000.

            # Print Info to File for Load Love Numbers
            out_head = ("../output/Love_Numbers/Partials/header_"+str(jj)+par_out)
            out_body = ("../output/Love_Numbers/Partials/body_"+str(jj)+par_out)
            out_name = ("../output/Love_Numbers/Partials/lln_n"+str(int(cn))+"_"+par_out)
            params = np.column_stack((rint,imnd,iknd,irnd,dhl_dmu[jj,:],dhl_dK[jj,:],dhl_drho[jj,:],dll_dmu[jj,:],dll_dK[jj,:],dll_drho[jj,:],\
                dkl_dmu[jj,:],dkl_dK[jj,:],dkl_drho[jj,:]))
            hf = open(out_head,'w')
            uh1 = ('------------------------------------------------------ \n')
            uh2 = (':: Load Love Number Partial Derivatives \n')
            uh3 = (':: Computed at ' + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + '\n')
            uh4 = ('------------------------------------------------------ \n')
            hf.write(uh1); hf.write(uh2); hf.write(uh3); hf.write(uh4)
            out_str1 = ("n = "+str(int(cn))+" | Normalization: mu,kappa = (R_sc*(L_sc**2)*(T_sc**(-2))); rho = R_sc \n")
            out_str2 = ("    L_sc = "+str(L_sc)+ " | R_sc = "+str(R_sc)+ " | T_sc = "+str(T_sc)+ " | pi*G = "+str(piG)+ "\n")
            out_str3 = ' Radius(km)  Mu[non-dim]  Kappa[non-dim]  Rho[non-dim]  dhl_dmu  dhl_dK  dhl_drho  dll_dmu  dll_dK  dll_drho  dkl_dmu  dkl_dK  dkl_drho \n'
            out_str4 = ' ************************************************************************************************************************************** \n'
            hf.write(out_str1); hf.write(out_str2); hf.write(out_str3)
            hf.close()
            np.savetxt(out_body,params,fmt='%f %f %f %f %f %f %f %f %f %f %f %f %f')
            filenames_lln = [out_head, out_body]
            with open(out_name,'w') as outfile:
                for fname in filenames_lln:
                    with open(fname) as infile:
                        outfile.write(infile.read())
            os.remove(out_head)
            os.remove(out_body)

            # Print Info to File for Potential/Tide Love Numbers
            out_head = ("../output/Love_Numbers/Partials/header_"+str(jj)+par_out)
            out_body = ("../output/Love_Numbers/Partials/body_"+str(jj)+par_out)
            out_name = ("../output/Love_Numbers/Partials/pln_n"+str(int(cn))+"_"+par_out)
            params = np.column_stack((rint,imnd,iknd,irnd,dht_dmu[jj,:],dht_dK[jj,:],dht_drho[jj,:],dlt_dmu[jj,:],dlt_dK[jj,:],dlt_drho[jj,:],\
                dkt_dmu[jj,:],dkt_dK[jj,:],dkt_drho[jj,:]))
            hf = open(out_head,'w')
            uh1 = ('------------------------------------------------------ \n')
            uh2 = (':: Potential Love Number Partial Derivatives \n')
            uh3 = (':: Computed at ' + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + '\n')
            uh4 = ('------------------------------------------------------ \n')
            hf.write(uh1); hf.write(uh2); hf.write(uh3); hf.write(uh4)
            out_str1 = ("n = "+str(int(cn))+" | Normalization: mu,kappa = (R_sc*(L_sc**2)*(T_sc**(-2))); rho = R_sc \n")
            out_str2 = ("    L_sc = "+str(L_sc)+ " | R_sc = "+str(R_sc)+ " | T_sc = "+str(T_sc)+ " | pi*G = "+str(piG)+ "\n")
            out_str3 = ' Radius(km)  Mu[non-dim]  Kappa[non-dim]  Rho[non-dim]  dhp_dmu  dhp_dK  dhp_drho  dlp_dmu  dlp_dK  dlp_drho  dkp_dmu  dkp_dK  dkp_drho \n'
            out_str4 = ' ************************************************************************************************************************************** \n'
            hf.write(out_str1); hf.write(out_str2); hf.write(out_str3)
            hf.close()
            np.savetxt(out_body,params,fmt='%f %f %f %f %f %f %f %f %f %f %f %f %f')
            filenames_lln = [out_head, out_body]
            with open(out_name,'w') as outfile:
                for fname in filenames_lln:
                    with open(fname) as infile:
                        outfile.write(infile.read())
            os.remove(out_head)
            os.remove(out_body)

            # Print Info to File for Shear Love Numbers (Stress Solution for n=1)
            out_head = ("../output/Love_Numbers/Partials/header_"+str(jj)+par_out)
            out_body = ("../output/Love_Numbers/Partials/body_"+str(jj)+par_out)
            out_name = ("../output/Love_Numbers/Partials/sln_n"+str(int(cn))+"_"+par_out)
            params = np.column_stack((rint,imnd,iknd,irnd,dhs_dmu[jj,:],dhs_dK[jj,:],dhs_drho[jj,:],dls_dmu[jj,:],dls_dK[jj,:],dls_drho[jj,:],\
                dks_dmu[jj,:],dks_dK[jj,:],dks_drho[jj,:]))
            hf = open(out_head,'w')
            uh1 = ('------------------------------------------------------ \n')
            uh2 = (':: Shear Love Number Partial Derivatives \n')
            uh3 = (':: Computed at ' + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + '\n')
            uh4 = ('------------------------------------------------------ \n')
            hf.write(uh1); hf.write(uh2); hf.write(uh3); hf.write(uh4)
            if (cn == 1):
                out_str1 = ("n = "+str(int(cn))+" | STRESS SOLUTION | Normalization: mu,kappa = (R_sc*(L_sc**2)*(T_sc**(-2))); rho = R_sc \n")
            else:
                out_str1 = ("n = "+str(int(cn))+" | Normalization: mu,kappa = (R_sc*(L_sc**2)*(T_sc**(-2))); rho = R_sc \n")
            out_str2 = ("    L_sc = "+str(L_sc)+ " | R_sc = "+str(R_sc)+ " | T_sc = "+str(T_sc)+ " | pi*G = "+str(piG)+ "\n")
            out_str3 = ' Radius(km)  Mu[non-dim]  Kappa[non-dim]  Rho[non-dim]  dhs_dmu  dhs_dK  dhs_drho  dls_dmu  dls_dK  dls_drho  dks_dmu  dks_dK  dks_drho \n'
            out_str4 = ' ************************************************************************************************************************************** \n'
            hf.write(out_str1); hf.write(out_str2); hf.write(out_str3)
            hf.close()
            np.savetxt(out_body,params,fmt='%f %f %f %f %f %f %f %f %f %f %f %f %f')
            filenames_lln = [out_head, out_body]
            with open(out_name,'w') as outfile:
                for fname in filenames_lln:
                    with open(fname) as infile:
                        outfile.write(infile.read())
            os.remove(out_head)
            os.remove(out_body)

        # Return Results 
        return dht_dmu,dlt_dmu,dkt_dmu,dht_dK,dlt_dK,dkt_dK,dht_drho,dlt_drho,dkt_drho,dhl_dmu,dll_dmu,dkl_dmu,dhl_dK,dll_dK,dkl_dK,\
            dhl_drho,dll_drho,dkl_drho,dhs_dmu,dls_dmu,dks_dmu,dhs_dK,dls_dK,dks_dK,dhs_drho,dls_drho,dks_drho,\

    else:
 
        return


