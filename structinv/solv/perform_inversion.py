# *********************************************************************
# FUNCTION TO INVERT SURFACE DISPLACEMENTS FOR STRUCTURE
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

import numpy as np
import scipy as sc
from scipy.sparse import linalg
import sys
import os
from structinv.utility import read_datafile
from structinv.utility import read_datafile_uonly
from structinv.utility import read_datafile_harmonic
from structinv.utility import read_datafile_uonly_harmonic
from structinv.utility import read_starting_model

def main(datafile,fid,startmod,design_matrix,sta_ids,sta_comp_ids,sta_comp_lat,sta_comp_lon,pert_rad_bot,pert_rad_top,pert_param,tikhonov,alpha,beta,uonly=False,inc_imag=False,pme=False,\
    lat_filter_flag=False,keep_south=True,lat_filter=50.):

    # Read the Datafile
    if uonly: 
        if inc_imag:
            #   Format: Station, Latitude[+N], Longitude[+E], Up-Amp[mm], Up-Pha[deg]
            sta,slat,slon,ure,uim = read_datafile_uonly_harmonic.main(datafile,pme=pme)
        else: 
            #   Format: Station, Latitude[+N], Longitude[+E], Up-Displacement[mm]
            sta,slat,slon,udisp = read_datafile_uonly.main(datafile)
    else:
        if inc_imag:
            #   Format: Station, Latitude[+N], Longitude[+E], East-Amp[mm], East-Pha[deg], North-Amp[mm], North-Pha[deg], Up-Amp[mm], Up-Pha[deg]
            sta,slat,slon,ere,eim,nre,nim,ure,uim = read_datafile_harmonic.main(datafile,pme=pme)
        else:
            #   Format: Station, Latitude[+N], Longitude[+E], East-Displacement[mm], North-Displacement[mm], Up-Displacement[mm]
            sta,slat,slon,edisp,ndisp,udisp = read_datafile.main(datafile)

    # Find and remove nans
    if inc_imag:
        ure_nan = np.isnan(ure)
        ure = ure[ure_nan == 0]
        uim = uim[ure_nan == 0]
        sta = sta[ure_nan == 0]
        slat = slat[ure_nan == 0]
        slon = slon[ure_nan == 0]
        if (uonly == False):
            ere = ere[ure_nan == 0]
            eim = eim[ure_nan == 0]
            nre = nre[ure_nan == 0]
            nim = nim[ure_nan == 0]
    else: 
        udisp_nan = np.isnan(udisp)
        udisp = udisp[udisp_nan == 0]
        sta = sta[udisp_nan == 0]
        slat = slat[udisp_nan == 0]
        slon = slon[udisp_nan == 0]
        if (uonly == False):
            edisp = edisp[udisp_nan == 0]
            ndisp = ndisp[udisp_nan == 0]

    # Filter data by station latitude, if desired
    if lat_filter_flag:
        if inc_imag:
            if keep_south:
                keepidx = np.where(slat <= lat_filter)[0]
            else:
                keepidx = np.where(slat >= lat_filter)[0]
            ure = ure[keepidx]
            uim = uim[keepidx]
            sta = sta[keepidx]
            slat = slat[keepidx]
            slon = slon[keepidx]
            if not uonly:
                ere = ere[keepidx]
                eim = eim[keepidx]
                nre = nre[keepidx]
                nim = nim[keepidx]
        else:
            if keep_south: 
                keepidx = np.where(slat <= lat_filter)[0]
            else:
                keepidx = np.where(slat >= lat_filter)[0]
            udisp = udisp[keepidx]
            sta = sta[keepidx]
            slat = slat[keepidx]
            slon = slon[keepidx]
            if not uonly:
                edisp = edisp[keepidx]
                ndisp = ndisp[keepidx]

    # Read the Starting Model
    if inc_imag:
        msta,mlat,mlon,mere,meim,mnre,mnim,mure,muim = read_starting_model.main(startmod,inc_imag=inc_imag)
    else:
        msta,mlat,mlon,medisp,mndisp,mudisp = read_starting_model.main(startmod,inc_imag=inc_imag)

    # Find Stations that Match in both the Data File and the Starting Model File
    #   Otherwise, Remove Rows from Starting Model where No Data Exists for that Station
    #   Otherwise, Remove Observations where No Starting Model Exists for that Station
    #   Also, ensure correct order for stations in Starting Model and in Data File
    common_sta,data_idx,sm_idx = np.intersect1d(sta,msta,return_indices=True)
    if (len(data_idx) != len(sm_idx)):
        sys.exit(':: Error: Mismatch in stations. [perform_inversion.py]')
    sta = sta[data_idx]
    slat = slat[data_idx]
    slon = slon[data_idx]
    msta = msta[sm_idx]
    mlat = mlat[sm_idx]
    mlon = mlon[sm_idx]
    if inc_imag: 
        ure = ure[data_idx]
        uim = uim[data_idx]
        mure = mure[sm_idx]
        muim = muim[sm_idx]
        if not uonly:
            ere = ere[data_idx]
            eim = eim[data_idx]
            nre = nre[data_idx]
            nim = nim[data_idx]
            mere = mere[sm_idx]
            meim = meim[sm_idx]
            mnre = mnre[sm_idx]
            mnim = mnim[sm_idx]
    else:
        udisp = udisp[data_idx]
        mudisp = mudisp[sm_idx]
        if not uonly:
            edisp = edisp[data_idx]
            ndisp = ndisp[data_idx]
            medisp = medisp[sm_idx]
            mndisp = mndisp[sm_idx]

    # Subtract model from the data to compute the data vector: d = Gm0 + [d(Gm)/d(m)]*m --> (d-Gm0) = [d(Gm)/d(m)] m
    if uonly: 
        if inc_imag: 
            dure = np.subtract(ure,mure)
            duim = np.subtract(uim,muim)        
        else:
            dudisp = np.subtract(udisp,mudisp)
    else: 
        if inc_imag:
            dere = np.subtract(ere,mere)
            deim = np.subtract(eim,meim)
            dnre = np.subtract(nre,mnre)
            dnim = np.subtract(nim,mnim)
            dure = np.subtract(ure,mure)
            duim = np.subtract(uim,muim)
        else:
            dedisp = np.subtract(edisp,medisp)
            dndisp = np.subtract(ndisp,mndisp)
            dudisp = np.subtract(udisp,mudisp)

    # Find Stations that Match Design Matrix File
    #   Otherwise, Remove Rows from Design Matrix where No Data Exists for that Station
    #   Otherwise, Remove Observations where No Design-Matrix Model Exists for that Station
    #   Also, ensure correct order for stations in Design Matrix and in Data Vector
    common_sta,data_idx,dm_idx = np.intersect1d(sta,sta_ids,return_indices=True)
    if (len(data_idx) != len(dm_idx)):
        sys.exit(':: Error: Mismatch in stations. [perform_inversion.py]')
    # Build G matrix and data vector differently depending on number of spatial components
    if uonly: # one spatial component
        if inc_imag: 
            # Initialize updated Design Matrix
            G = np.empty((len(dm_idx)*2,len(pert_param))) # x2 because of real and imaginary
            # Initialize datavector
            d = np.empty((len(data_idx)*2,)) # x2 because of real and imaginary
            for bb in range(0,len(dm_idx)):
                crow_dm = dm_idx[bb]
                crow_data = data_idx[bb]
                # For harmonic loads, the order of the design matrix is ere,nre,ure,eim,nim,uim
                # up (note: it is assumed that the Design Matrix was built for 3 components -- e,n,u; hence, the +2)
                G[(bb*2),:] = design_matrix[crow_dm+2,:] # up, real
                G[(bb*2)+1,:] = design_matrix[crow_dm+5,:] # up, imaginary
                d[(bb*2)] = dure[crow_data] # up, real
                d[(bb*2)+1] = duim[crow_data] # up, imaginary
        else:
            # Initialize updated Design Matrix
            G = np.empty((len(dm_idx),len(pert_param)))
            # Initialize datavector
            d = np.empty((len(data_idx),))
            for bb in range(0,len(dm_idx)):
                crow_dm = dm_idx[bb]
                crow_data = data_idx[bb]
                G[bb,:] = design_matrix[crow_dm+2,:] # up (note: it is assumed that the Design Matrix was built for 3 components -- e,n,u; hence, the +2)
                d[bb] = dudisp[crow_data] # up
    else: # three components (e,n,u)
        if inc_imag:
            # Initialize updated Design Matrix
            G = np.empty((len(dm_idx)*6,len(pert_param))) # x6 because of real and imaginary and three spatial components (e,n,u) for each
            # Initialize datavector
            d = np.empty((len(data_idx)*6,)) # x6 because of real and imaginary and three spatial components (e,n,u) for each
            for bb in range(0,len(dm_idx)):
                crow_dm = dm_idx[bb]
                crow_data = data_idx[bb]
                G[(bb*6),:] = design_matrix[crow_dm,:] # east, real
                G[(bb*6)+1,:] = design_matrix[crow_dm+1,:] # north, real
                G[(bb*6)+2,:] = design_matrix[crow_dm+2,:] # up, real
                G[(bb*6)+3,:] = design_matrix[crow_dm+3,:] # east, imag
                G[(bb*6)+4,:] = design_matrix[crow_dm+4,:] # north, imag
                G[(bb*6)+5,:] = design_matrix[crow_dm+5,:] # up, imag
                d[(bb*6)] = dere[crow_data] # east, real
                d[(bb*6)+1] = dnre[crow_data] # north, real
                d[(bb*6)+2] = dure[crow_data] # up, real
                d[(bb*6)+3] = deim[crow_data] # east, imag
                d[(bb*6)+4] = dnim[crow_data] # north, imag
                d[(bb*6)+5] = duim[crow_data] # up, imag
        else:
            # Initialize updated Design Matrix
            G = np.empty((len(dm_idx)*3,len(pert_param)))
            # Initialize datavector
            d = np.empty((len(data_idx)*3,))
            for bb in range(0,len(dm_idx)):
                crow_dm = dm_idx[bb]
                crow_data = data_idx[bb]
                G[(bb*3),:] = design_matrix[crow_dm,:] # east
                G[(bb*3)+1,:] = design_matrix[crow_dm+1,:] # north
                G[(bb*3)+2,:] = design_matrix[crow_dm+2,:] # up
                d[(bb*3)] = edisp[crow_data] # east
                d[(bb*3)+1] = ndisp[crow_data] # north
                d[(bb*3)+2] = udisp[crow_data] # up    

    # Bias the Solution with Tikhonov Regularization
    if (tikhonov == 'zeroth'): # Zeroth-Order Tikhonov Regularization
        # Create identity matrix 
        L0 = np.identity(len(pert_param))
        # ** Solve by redefining the design matrix and data vector to include regularization (Aster, Borchers, & Thurber [2013], Eq. 4.5)
        C = np.concatenate((G,(beta*L0)),axis=0)
        zero_vec = np.zeros(len(load_cell_ids))
        b = np.concatenate((d,zero_vec))

    elif (tikhonov == 'second' or tikhonov == 'zeroth_second'): # Second-Order Tikhonov Regularization (in one dimension)
        # Create L matrix (see Aster, Borchers, & Thurber (2013), Chapter 4, Section 4.4: Higher Order Tikhonov Regularization)
        L = []
        for cc in range(0,len(pert_param)):
            # Current parameter
            cparam = pert_param[cc]
            # Current bottom radius
            cbr = pert_rad_bot[cc]
            # Current top radius
            ctr = pert_rad_top[cc]
            # Find neighboring layers (for the same material parameter)
            upper = np.where((pert_rad_bot == ctr) & (pert_param == cparam)); upper = upper[0]
            lower = np.where((pert_rad_top == cbr) & (pert_param == cparam)); lower = lower[0]
            crow = np.zeros((len(pert_param),))
            crow[cc] = -2
            if not upper: # no upper neighbor
                crow[cc] += 1
            else:
                crow[upper] = 1
            if not lower: # no lower neighbor
                crow[cc] += 1
            else:
                crow[lower] = 1
            L.append(crow)
        L2 = np.asarray(L)
        # ** Solve by redefining the design matrix and data vector to include regularization (Aster, Borchers, & Thurber [2013], Eq. 4.5)
        C = np.concatenate((G,(alpha*L2)),axis=0)
        zero_vec_rows = L2.shape[0]
        zero_vec = np.zeros(zero_vec_rows)
        b = np.concatenate((d,zero_vec))
        # Combined zeroth and second order
        if (tikhonov == 'zeroth_second'): 
            L0 = np.identity(len(pert_param))
            # ** Solve by redefining the design matrix and data vector to include regularization (Aster, Borchers, & Thurber [2013], Eq. 4.5)
            C = np.concatenate((C,(beta*L0)),axis=0)
            zero_vec = np.zeros(len(load_cell_ids))
            b = np.concatenate((b,zero_vec))
 
    elif (tikhonov == 'none'): # No regularization
        C = G.copy()
        b = d.copy()

    else: 
        sys.exit(':: Error: Invalid Tikhonov regularization code. [perform_inversion.py]')

    # SOLVE (Conjugate Gradient)
    #print(C)
    #print(b)
    CTC = np.dot(C.T,C)
    CTb = np.dot(C.T,b)    
    mvec,info = linalg.cg(CTC, CTb)

    # Return Model Vector
    return mvec



# ----------------------------------------------------------------------------------------------------------------
# Alternative choices for solving the system
# Option A: Solve by redefining the design matrix and data vector to include regularization (Aster, Borchers, & Thurber [2013], Eq. 4.26)
#  Then use SVD, but could also use a range of other methods.
#C = np.concatenate((G,(alpha*L)),axis=0)
#zero_vec_rows = L.shape[0]
#zero_vec = np.zeros(zero_vec_rows)
#b = np.concatenate((d,zero_vec))
#U, S, V = np.linalg.svd(C, full_matrices=False)
#mvec = np.dot(V.T, np.linalg.solve(np.diag(S), np.dot(U.T, b)))
#print(mvec)
#print(min(mvec))
#print(max(mvec))
# Compute (L^T L)
#LTL = np.dot(L.T,L)
# Solve the inverse problem (e.g., Aster, Borchers, & Thurber (2013), Eq. 4.43)
# Option B: Simple solution (with matrix inversion) [APPEARS TO BE UNSTABLE]
#mvec = np.dot(np.dot(np.linalg.inv(np.dot(G.T, G) + (alpha**2)*LTL), G.T), d)
#print(mvec)
#print(min(mvec))
#print(max(mvec))
# Option C: Solve the system of equations directly
#mvec = np.linalg.solve((np.dot(G.T, G) + (alpha**2)*LTL), np.dot(G.T, d))
#print(mvec)
#print(min(mvec))
#print(max(mvec))
# Option D: Solve the system with NumPy least-squares function
#mvec = np.linalg.lstsq((np.dot(G.T, G) + (alpha**2)*LTL), np.dot(G.T, d), rcond=None)
#mvec = mvec[0]
#print(mvec)
#print(min(mvec))
#print(max(mvec))
# Option E: Conjugate Gradient (requires a square matrix)
#CTC = np.dot(C.T,C)
#CTb = np.dot(C.T,b)
#mvec,info = linalg.cg(CTC, CTb)
#print(mvec)
#print(min(mvec))
#print(max(mvec))
# Option F: Solve with nnls in scipy.optimize
#mvec,rnorm = sc.optimize.nnls(C,b)
#print(mvec)
# Option F [TESTING]: Solve using singular value decomposition (SVD) with filter factors
#   See Aster, Borchers, & Thurber (2013), Section 4.2
#U, S, V = np.linalg.svd(G, full_matrices=False)
#Sfull = np.diag(S)
#UT = np.transpose(U)
#VT = np.transpose(V)
#I = np.identity(len(S))
#fi = np.divide( np.square(S) , (np.square(S) + alpha**2) ) # filter factors: Aster, Borchers, Thurber (2013), Eq. 4.17
#xi = np.empty((len(S),))
#for mm in range(0,len(S)):
#    xi[mm] = np.multiply(S[mm], np.dot(UT[:,mm],d)) / (np.square(S[mm]) + alpha**2) # Aster, Borchers, Thurber (2013), Eq. 4.13
#xvec = np.linalg.solve((np.dot(Sfull.T, Sfull) + (alpha**2)*I), np.dot( np.dot(Sfull.T, UT) , d))
#print(xi)


