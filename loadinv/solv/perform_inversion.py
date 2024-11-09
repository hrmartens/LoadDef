# *********************************************************************
# FUNCTION TO INVERT SURFACE DISPLACEMENTS FOR SURFACE LOAD
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
from loadinv.utility import read_datafile
from loadinv.utility import read_datafile_uonly

def main(datafile,fid,design_matrix,sta_ids,sta_comp_ids,sta_comp_lat,sta_comp_lon,load_cell_ids,load_cell_lat,load_cell_lon,tikhonov,alpha,beta,nine_point=True,uonly=False):

    # Read the Datafile
    if (uonly == True): 
        #   Format: Station, Latitude[+N], Longitude[+E], Up-Displacement[mm]
        sta,slat,slon,udisp = read_datafile_uonly.main(datafile)
    else:
        #   Format: Station, Latitude[+N], Longitude[+E], East-Displacement[mm], North-Displacement[mm], Up-Displacement[mm]
        sta,slat,slon,edisp,ndisp,udisp = read_datafile.main(datafile)

    # Find Stations that Match Design Matrix File
    #   Otherwise, Remove Rows from Design Matrix where No Data Exists for that Station
    #   Otherwise, Remove Observations where No Design Matrix Model Exists for that Station
    #   Also, ensure correct order for stations in Design Matrix and in Data Vector
    common_sta,data_idx,dm_idx = np.intersect1d(sta,sta_ids,return_indices=True)
    if (len(data_idx) != len(dm_idx)):
        sys.exit(':: Error: Mismatch in stations. [perform_inversion.py]')
    # Build G matrix and data vector differently depending on number of components
    if (uonly == True): # one component
        # Initialize updated Design Matrix 
        G = np.empty((len(dm_idx),len(load_cell_ids)))
        # Initialize datavector
        d = np.empty((len(data_idx),))
        for bb in range(0,len(dm_idx)):
            crow_dm = dm_idx[bb]
            crow_data = data_idx[bb]
            G[bb,:] = design_matrix[crow_dm+2,:] # up (note: it is assumed that the Design Matrix was built for 3 components -- e,n,u; hence, the +2)
            d[bb] = udisp[crow_data] # up
    else: # three components (e,n,u)
        # Initialize updated Design Matrix 
        G = np.empty((len(dm_idx)*3,len(load_cell_ids)))
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
        L0 = np.identity(len(load_cell_ids))
        # ** Solve by redefining the design matrix and data vector to include regularization (Aster, Borchers, & Thurber [2013], Eq. 4.5)
        C = np.concatenate((G,(beta*L0)),axis=0)
        zero_vec = np.zeros(len(load_cell_ids))
        b = np.concatenate((d,zero_vec))
        # ** Solve with Conjugate Gradient (requires a square matrix)
        CTC = np.dot(C.T,C)
        CTb = np.dot(C.T,b)
        mvec,info = linalg.cg(CTC, CTb)
        #print(mvec)
        #print(min(mvec))
        #print(max(mvec))

    elif (tikhonov == 'second' or tikhonov == 'zeroth_second'): # Second-Order Tikhonov Regularization (in two dimensions)            
        # Search for load cells that are surrounded on all four sides by other load cells (n,e,w,s)
        coordinates = np.empty((len(load_cell_lat),2))
        coordinates[:,0] = load_cell_lat
        coordinates[:,1] = load_cell_lon
        unq_lat = np.unique(load_cell_lat)
        unq_lon = np.unique(load_cell_lon)
        lat_sep = (max(unq_lat) - min(unq_lat)) / (len(unq_lat)-1)
        lon_sep = (max(unq_lon) - min(unq_lon)) / (len(unq_lon)-1)
        print(':: Note: Computing longitude separation value and neareset neighbors may fail if working across the Prime Meridian. Please use caution and verify results. [perform_inversion.py]')
        print(':: Note: For small or unusual cell separation distances, you may need to verify that cells are being found using numpy.where (tolerance of zero). [perform_inversion.py]')
        # Initialize L matrix (see Aster, Borchers, & Thurber (2013), Chapter 4, Exercise 4.3)
        L2 = []
        for cc in range(0,len(load_cell_ids)):
            ccell_id = load_cell_ids[cc]
            ccelllat = load_cell_lat[cc]
            ccelllon = load_cell_lon[cc]
            # Search for nearest neighbors (south, north, west, and east)
            nn_slat = ccelllat - lat_sep
            nn_nlat = ccelllat + lat_sep 
            nn_wlon = ccelllon - lon_sep
            nn_elon = ccelllon + lon_sep
            nn_south = np.where((load_cell_lat == nn_slat) & (load_cell_lon == ccelllon)); nn_south = nn_south[0]
            nn_north = np.where((load_cell_lat == nn_nlat) & (load_cell_lon == ccelllon)); nn_north = nn_north[0]
            nn_west = np.where((load_cell_lon == nn_wlon) & (load_cell_lat == ccelllat)); nn_west = nn_west[0]
            nn_east = np.where((load_cell_lon == nn_elon) & (load_cell_lat == ccelllat)); nn_east = nn_east[0]
            if (nine_point == True): # search for southwest, southeast, northwest, and northeast points (e.g., Rosser 1975 Comp. & Maths. with Appls.)
                # Rosser, J. Barkley (1975). "Nine-point difference solutions for Poisson's equation," Computations and Mathematics with Applications, Vol. 1, 351-360.
                # See https://www.sciencedirect.com/science/article/pii/0898122175900358
                nn_sw = np.where((load_cell_lat == nn_slat) & (load_cell_lon == nn_wlon)); nn_sw = nn_sw[0]
                nn_se = np.where((load_cell_lat == nn_slat) & (load_cell_lon == nn_elon)); nn_se = nn_se[0]
                nn_nw = np.where((load_cell_lat == nn_nlat) & (load_cell_lon == nn_wlon)); nn_nw = nn_nw[0]
                nn_ne = np.where((load_cell_lat == nn_nlat) & (load_cell_lon == nn_elon)); nn_ne = nn_ne[0]
                # Current cell has all eight neighbors; add to L matrix
                # See Aster, Borchers, & Thurber (2013), Exercise 4.3 (Section 4.9, Page 126) for an example of how to
                #   generate the L matrix for second-order Tikhonov regularization in two dimensions
                # See Rosser (1975) Comp. & Maths. with Appls., Vol 1, pp. 351-360 for 9-point stencil
                #  Rosser, J. Barkley (1975). "Nine-point difference solutions for Poisson's equation," Computations and Mathematics with Applications, Vol. 1, 351-360.
                crow = np.zeros((len(load_cell_ids),))
                crow[cc] = -20
                if not nn_south: # no south neighbor
                    crow[cc] += 4
                else:
                    crow[nn_south] = 4
                if not nn_north: # no north neighbor
                    crow[cc] += 4
                else:
                    crow[nn_north] = 4
                if not nn_west: # no west neighbor
                    crow[cc] += 4
                else:
                    crow[nn_west] = 4
                if not nn_east: # no east neighbor
                    crow[cc] += 4
                else:
                    crow[nn_east] = 4
                if not nn_sw: # no southwest neighbor
                    crow[cc] += 1
                else:
                    crow[nn_sw] = 1
                if not nn_se: # no southeast neighbor
                    crow[cc] += 1
                else:
                    crow[nn_se] = 1
                if not nn_nw: # no northwest neighbor
                    crow[cc] += 1
                else:
                    crow[nn_nw] = 1
                if not nn_ne: # no northeast neighbor
                    crow[cc] += 1
                else:
                    crow[nn_ne] = 1
                L2.append(crow)
            else:
                # Current cell has all four neighbors; add to L matrix
                # See Aster, Borchers, & Thurber (2013), Exercise 4.3 (Section 4.9, Page 126) for an example of how to 
                #   generate the L matrix for second-order Tikhonov regularization in two dimensions
                crow = np.zeros((len(load_cell_ids),))
                crow[cc] = -4
                if not nn_south: # no south neighbor
                    crow[cc] += 1
                else: 
                    crow[nn_south] = 1
                if not nn_north: # no north neighbor
                    crow[cc] += 1
                else:
                    crow[nn_north] = 1
                if not nn_west: # no west neighbor
                    crow[cc] += 1
                else:
                    crow[nn_west] = 1
                if not nn_east: # no east neighbor
                    crow[cc] += 1
                else:
                    crow[nn_east] = 1
                L2.append(crow)
        L2 = np.asarray(L2)
        # ** Solve by redefining the design matrix and data vector to include regularization (Aster, Borchers, & Thurber [2013], Eq. 4.26)
        C = np.concatenate((G,(alpha*L2)),axis=0)
        zero_vec_rows = L2.shape[0]
        zero_vec = np.zeros(zero_vec_rows)
        b = np.concatenate((d,zero_vec))

        # Second Order Only
        if (tikhonov == 'second'): 
            # Solve with Conjugate Gradient (requires a square matrix)
            CTC = np.dot(C.T,C)
            CTb = np.dot(C.T,b)
            mvec,info = linalg.cg(CTC, CTb)
            #print(mvec)
            #print(min(mvec))
            #print(max(mvec))

        # Zeroth and Second Order Combined
        elif (tikhonov == 'zeroth_second'):
            # Create identity matrix
            L0 = np.identity(len(load_cell_ids))
            # ** Solve by redefining the design matrix and data vector to include regularization (Aster, Borchers, & Thurber [2013], Eq. 4.5)
            C = np.concatenate((C,(beta*L0)),axis=0)
            zero_vec = np.zeros(len(load_cell_ids))
            b = np.concatenate((b,zero_vec))
            # Solve with Conjugate Gradient (requires a square matrix)
            CTC = np.dot(C.T,C)
            CTb = np.dot(C.T,b)
            mvec,info = linalg.cg(CTC, CTb)
            #print(mvec)
            #print(min(mvec))
            #print(max(mvec))

    elif (tikhonov == 'none'):
        # No regularization
        # Solve with Conjugate Gradient (requires a square matrix)
        CTC = np.dot(G.T,G)
        CTb = np.dot(G.T,d)
        mvec,info = linalg.cg(CTC, CTb)
        #print(mvec)
        #print(min(mvec))
        #print(max(mvec))

    else:
        sys.exit(':: Error: Invalid Tikhonov regularization code. [perform_inversion.py]')

    # Return Model Vector
    return mvec

