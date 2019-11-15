# *********************************************************************
# CAUTION: IN DEVELOPMENT!!
#
# FUNCTION TO COMPUTE THE ASYMPTOTIC PARTIAL DERIVATIVES OF LOVE NUMBERS
# SEE: Okubo (1988a), Geophysical Journal, 92, p. 39-51
#               Asymptotic solutions to the static deformation of the Earth - I. Spheroidal mode
#             Okubo (1988b), Bureau Gravimetrique International, Bulletin D'Information, No. 62
#               Green's function to a point load - its dependence on Earth model
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
from LOADGF.PL import ln_partials_asymptotic
from LOADGF.LN import prepare_earth_model
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.integrate import simps
 
"""
Function to compute asymptotic partial derivatives of Love numbers. 
For more information on the theory, see Okubo & Saito (1983), Martens et al. (2016), 
    Okubo (1988a), Okubo (1988b)

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

par_out : Extension for the output filenames.
    Default is "partials.txt"
"""

def main(myn,sint,Yload,Ypot,Yshr,Ystr,hload,nlload,nkload,hpot,nlpot,nkpot,hshr,nlshr,nkshr,\
    hstr,nlstr,nkstr,emod_file,rank,comm,size,par_out='partials.txt',G=6.627E-11,r_min=1000.,kx=1,period_hours=12.42,delim=None):

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
        print(":: Computing Asymptotic Partial Derivatives. Please Wait..."); print("")

        # For SNREI Earth, Angular Frequency (omega) is Zero 
        # Azimuthal order is only utilized for a rotating Earth
        # The variables for each are included here as "place-holders" for future versions
        omega = 0
        order = 2

        # Prepare the Earth Model (read in, non-dimensionalize elastic parameters, etc.)
        r,mu,K,lmda,rho,g,tck_lnd,tck_mnd,tck_rnd,tck_gnd,s,lnd,mnd,rnd,gnd,s_min,small,\
            earth_radius,earth_mass,sic,soc,adim,gsdim,pi,piG,L_sc,R_sc,T_sc = \
            prepare_earth_model.main(emod_file,G,r_min,kx,file_delim=delim)

        # Define Forcing Period
        w = (1./(period_hours*3600.))*(2.*pi) # convert period to frequency (rad/sec)
        wnd = w*T_sc                         # non-dimensionalize
        ond = omega*T_sc

        # Initialize Arrays of Asymptotic Partial Derivatives
        adht_dmu  = np.empty((len(myn),sint_length))
        adlt_dmu  = np.empty((len(myn),sint_length))
        adkt_dmu  = np.empty((len(myn),sint_length))
        adht_dK   = np.empty((len(myn),sint_length))
        adlt_dK   = np.empty((len(myn),sint_length))
        adkt_dK   = np.empty((len(myn),sint_length))
        adht_drho = np.empty((len(myn),sint_length))
        adlt_drho = np.empty((len(myn),sint_length))
        adkt_drho = np.empty((len(myn),sint_length))
        adhl_dmu  = np.empty((len(myn),sint_length))
        adll_dmu  = np.empty((len(myn),sint_length))
        adkl_dmu  = np.empty((len(myn),sint_length))
        adhl_dK   = np.empty((len(myn),sint_length))
        adll_dK   = np.empty((len(myn),sint_length))
        adkl_dK   = np.empty((len(myn),sint_length))
        adhl_drho = np.empty((len(myn),sint_length))
        adll_drho = np.empty((len(myn),sint_length))
        adkl_drho = np.empty((len(myn),sint_length))
        adhs_dmu  = np.empty((len(myn),sint_length))
        adls_dmu  = np.empty((len(myn),sint_length))
        adks_dmu  = np.empty((len(myn),sint_length))
        adhs_dK   = np.empty((len(myn),sint_length))
        adls_dK   = np.empty((len(myn),sint_length))
        adks_dK   = np.empty((len(myn),sint_length))
        adhs_drho = np.empty((len(myn),sint_length))
        adls_drho = np.empty((len(myn),sint_length))
        adks_drho = np.empty((len(myn),sint_length)) 

    else: 

        # Workers Know Nothing About the Data
        adht_dmu = adlt_dmu = adkt_dmu = adht_dK = adlt_dK = adkt_dK = adht_drho = adlt_drho = adkt_drho = None
        adhl_dmu = adll_dmu = adkl_dmu = adhl_dK = adll_dK = adkl_dK = adhl_drho = adll_drho = adkl_drho = None
        adhs_dmu = adls_dmu = adks_dmu = adhs_dK = adls_dK = adks_dK = adhs_drho = adls_drho = adks_drho = None
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
    adht_dmu_sub  = np.empty((len(n_sub),sint_length))
    adlt_dmu_sub  = np.empty((len(n_sub),sint_length))
    adkt_dmu_sub  = np.empty((len(n_sub),sint_length))
    adht_dK_sub   = np.empty((len(n_sub),sint_length))
    adlt_dK_sub   = np.empty((len(n_sub),sint_length))
    adkt_dK_sub   = np.empty((len(n_sub),sint_length))
    adht_drho_sub = np.empty((len(n_sub),sint_length))
    adlt_drho_sub = np.empty((len(n_sub),sint_length))
    adkt_drho_sub = np.empty((len(n_sub),sint_length))
    adhl_dmu_sub  = np.empty((len(n_sub),sint_length))
    adll_dmu_sub  = np.empty((len(n_sub),sint_length))
    adkl_dmu_sub  = np.empty((len(n_sub),sint_length))
    adhl_dK_sub   = np.empty((len(n_sub),sint_length))
    adll_dK_sub   = np.empty((len(n_sub),sint_length))
    adkl_dK_sub   = np.empty((len(n_sub),sint_length))
    adhl_drho_sub = np.empty((len(n_sub),sint_length))
    adll_drho_sub = np.empty((len(n_sub),sint_length))
    adkl_drho_sub = np.empty((len(n_sub),sint_length))
    adhs_dmu_sub  = np.empty((len(n_sub),sint_length))
    adls_dmu_sub  = np.empty((len(n_sub),sint_length))
    adks_dmu_sub  = np.empty((len(n_sub),sint_length))
    adhs_dK_sub   = np.empty((len(n_sub),sint_length))
    adls_dK_sub   = np.empty((len(n_sub),sint_length))
    adks_dK_sub   = np.empty((len(n_sub),sint_length))
    adhs_drho_sub = np.empty((len(n_sub),sint_length))
    adls_drho_sub = np.empty((len(n_sub),sint_length))
    adks_drho_sub = np.empty((len(n_sub),sint_length))
    for ii in range(0,len(n_sub)):
        current_n = n_sub[ii]
        print('Partial Derivatives | Working on Harmonic Degree: %7s | Number: %6d of %6d | Rank: %6d' %(str(int(current_n)), (ii+1), len(n_sub), rank))
        nidx = int(current_n - min(myn)) # Determine Index (If the Range of 'n' Does Not Start at Zero)
        # Compute Asymptotic Results for the Current Spherical Harmonic Degree, n
        adht_dmu_sub[ii,:],adlt_dmu_sub[ii,:],adkt_dmu_sub[ii,:],adht_dK_sub[ii,:],adlt_dK_sub[ii,:],adkt_dK_sub[ii,:],adht_drho_sub[ii,:],adlt_drho_sub[ii,:],adkt_drho_sub[ii,:],\
            adhl_dmu_sub[ii,:],adll_dmu_sub[ii,:],adkl_dmu_sub[ii,:],adhl_dK_sub[ii,:],adll_dK_sub[ii,:],adkl_dK_sub[ii,:],adhl_drho_sub[ii,:],adll_drho_sub[ii,:],adkl_drho_sub[ii,:],\
            adhs_dmu_sub[ii,:],adls_dmu_sub[ii,:],adks_dmu_sub[ii,:],adhs_dK_sub[ii,:],adls_dK_sub[ii,:],adks_dK_sub[ii,:],adhs_drho_sub[ii,:],adls_drho_sub[ii,:],adks_drho_sub[ii,:] = \
            ln_partials_asymptotic.main(current_n,sint[nidx,:],hload[nidx],nlload[nidx],nkload[nidx],hpot[nidx],nlpot[nidx],nkpot[nidx],hshr[nidx],\
            nlshr[nidx],nkshr[nidx],hstr[nidx],nlstr[nidx],nkstr[nidx],tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,order)

    # Gather Results    
    comm.Gatherv(adht_dmu_sub, [adht_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adlt_dmu_sub, [adlt_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adkt_dmu_sub, [adkt_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adht_dK_sub, [adht_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adlt_dK_sub, [adlt_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adkt_dK_sub, [adkt_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adht_drho_sub, [adht_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adlt_drho_sub, [adlt_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adkt_drho_sub, [adkt_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adhl_dmu_sub, [adhl_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adll_dmu_sub, [adll_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adkl_dmu_sub, [adkl_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adhl_dK_sub, [adhl_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adll_dK_sub, [adll_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adkl_dK_sub, [adkl_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adhl_drho_sub, [adhl_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adll_drho_sub, [adll_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adkl_drho_sub, [adkl_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adhs_dmu_sub, [adhs_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adls_dmu_sub, [adls_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adks_dmu_sub, [adks_dmu, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adhs_dK_sub, [adhs_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adls_dK_sub, [adls_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adks_dK_sub, [adks_dK, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adhs_drho_sub, [adhs_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adls_drho_sub, [adls_drho, (sendcounts, None), ptype], root=0)
    comm.Gatherv(adks_drho_sub, [adks_drho, (sendcounts, None), ptype], root=0)

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
            out_name = ("../output/Love_Numbers/Partials/lln_asymptotic_n"+str(int(cn))+"_"+par_out)
            params = np.column_stack((rint,imnd,iknd,irnd,adhl_dmu[jj,:],adhl_dK[jj,:],adhl_drho[jj,:],adll_dmu[jj,:],adll_dK[jj,:],adll_drho[jj,:],\
                adkl_dmu[jj,:],adkl_dK[jj,:],adkl_drho[jj,:]))
            hf = open(out_head,'w')
            uh1 = ('------------------------------------------------------ \n')
            uh2 = (':: Asymptotic Love Number Partial Derivatives \n')
            uh3 = (':: Computed at ' + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + '\n')
            uh4 = ('------------------------------------------------------ \n')
            hf.write(uh1); hf.write(uh2); hf.write(uh3); hf.write(uh4)
            out_str1 = ("n = "+str(int(cn))+" | Normalization: mu,kappa = (R_sc*(L_sc**2)*(T_sc**(-2))); rho = R_sc \n")
            out_str2 = ("    L_sc = "+str(L_sc)+ " | R_sc = "+str(R_sc)+ " | T_sc = "+str(T_sc)+ " | pi*G = "+str(piG)+ "\n")
            out_str3 = ' Radius(km)  Mu[non-dim]  Kappa[non-dim]  Rho[non-dim]  dhl_dmu  dhl_dK  dhl_drho  dll_dmu  dll_dK  dll_drho dkl_dmu  dkl_dK  dkl_drho \n'
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
            out_name = ("../output/Love_Numbers/Partials/pln_asymptotic_n"+str(int(cn))+"_"+par_out)
            params = np.column_stack((rint,imnd,iknd,irnd,adht_dmu[jj,:],adht_dK[jj,:],adht_drho[jj,:],adlt_dmu[jj,:],adlt_dK[jj,:],adlt_drho[jj,:],\
                adkt_dmu[jj,:],adkt_dK[jj,:],adkt_drho[jj,:]))
            hf = open(out_head,'w')
            uh1 = ('------------------------------------------------------ \n')
            uh2 = (':: Asymptotic Love Number Partial Derivatives \n')
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
            out_name = ("../output/Love_Numbers/Partials/sln_asymptotic_n"+str(int(cn))+"_"+par_out)
            params = np.column_stack((rint,imnd,iknd,irnd,adhs_dmu[jj,:],adhs_dK[jj,:],adhs_drho[jj,:],adls_dmu[jj,:],adls_dK[jj,:],adls_drho[jj,:],\
                adks_dmu[jj,:],adks_dK[jj,:],adks_drho[jj,:]))
            hf = open(out_head,'w')
            uh1 = ('------------------------------------------------------ \n')
            uh2 = (':: Asymptotic Love Number Partial Derivatives \n')
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
        return adht_dmu,adlt_dmu,adkt_dmu,adht_dK,adlt_dK,adkt_dK,adht_drho,adlt_drho,adkt_drho,adhl_dmu,adll_dmu,adkl_dmu,adhl_dK,adll_dK,adkl_dK,\
            adhl_drho,adll_drho,adkl_drho,adhs_dmu,adls_dmu,adks_dmu,adhs_dK,adls_dK,adks_dK,adhs_drho,adls_drho,adks_drho

    else:
 
        return


