# *********************************************************************
# PURPOSE: Compute Residuals Between Two Sets of PMEs
# LITERATURE: Martens et al. (2016, GJI)
# 
# Copyright (c) 2014-2022: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
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

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd() + "/../../")

# IMPORT ADDITIONAL MODULES
from CONVGF.utility import env2pme
from CONVGF.utility import read_pme_file
import numpy as np
from math import pi

#* Compute the residuals between two sets of PME predictions/observations *#

# Main Function
def main(filename1,filename2,myoutfile,rmCMode,stations_to_exclude):

    # Read In Files 
    sta1,lat1,lon1,edir,smmj,smmn,eamp1,epha1,namp1,npha1,vamp1,vpha1 = read_pme_file.main(filename1)
    sta2,lat2,lon2,edir,smmj,smmn,eamp2,epha2,namp2,npha2,vamp2,vpha2 = read_pme_file.main(filename2)

    # Ensure that Station Names match
    #  1. Sort Stations Alphabetically
    unique_sta1, sta1idx = np.unique(sta1,return_index=True)
    sta1 = sta1[sta1idx]; lat1 = lat1[sta1idx]; lon1 = lon1[sta1idx]; eamp1 = eamp1[sta1idx]
    epha1 = epha1[sta1idx]; namp1 = namp1[sta1idx]
    npha1 = npha1[sta1idx]; vamp1 = vamp1[sta1idx]; vpha1 = vpha1[sta1idx]
    unique_sta2, sta2idx = np.unique(sta2,return_index=True)
    sta2 = sta2[sta2idx]; lat2 = lat2[sta2idx]; lon2 = lon2[sta2idx]; eamp2 = eamp2[sta2idx]
    epha2 = epha2[sta2idx]; namp2 = namp2[sta2idx]
    npha2 = npha2[sta2idx]; vamp2 = vamp2[sta2idx]; vpha2 = vpha2[sta2idx]
    #  2. Find Matching Entries
    sta2idx = [1 if i in sta1 else 0 for i in sta2]; sta2idx = np.nonzero(sta2idx); sta2idx = sta2idx[0]
    sta2 = sta2[sta2idx]; lat2 = lat2[sta2idx]; lon2 = lon2[sta2idx]; eamp2 = eamp2[sta2idx] 
    epha2 = epha2[sta2idx]; namp2 = namp2[sta2idx]
    npha2 = npha2[sta2idx]; vamp2 = vamp2[sta2idx]; vpha2 = vpha2[sta2idx]
    sta1idx = [1 if i in sta2 else 0 for i in sta1]; sta1idx = np.nonzero(sta1idx); sta1idx = sta1idx[0]
    sta1 = sta1[sta1idx]; lat1 = lat1[sta1idx]; lon1 = lon1[sta1idx]; eamp1 = eamp1[sta1idx]
    epha1 = epha1[sta1idx]; namp1 = namp1[sta1idx]
    npha1 = npha1[sta1idx]; vamp1 = vamp1[sta1idx]; vpha1 = vpha1[sta1idx]
    #  3. Do all Station Names now Match Up? Final Verification...
    if (set(sta1) == set(sta2)):
        sta = sta1
        lat = lat1
        lon = lon1
    else:
        sys.exit('Error: Station Arrays Do Not Match!')

    # Search for and Remove any NaNs
    idx_to_delete = []
    for jj in range(0,len(eamp1)):
        # Current Station
        ceamp = eamp1[jj]
        # Just Test One Component for NaN (All Components Should be NaN If One Component Is)
        test = np.isnan(ceamp)
        if test:
            idx_to_delete.append(jj)
    # Gather All Indices Together into an Array, and Apply to Station Data
    idx_to_delete = np.asarray(idx_to_delete)
    if (len(idx_to_delete) > 0):
        sta = np.delete(sta,idx_to_delete)
        lat = np.delete(lat,idx_to_delete)
        lon = np.delete(lon,idx_to_delete)
        eamp1 = np.delete(eamp1,idx_to_delete)
        epha1 = np.delete(epha1,idx_to_delete)
        namp1 = np.delete(namp1,idx_to_delete)
        npha1 = np.delete(npha1,idx_to_delete)
        vamp1 = np.delete(vamp1,idx_to_delete)
        vpha1 = np.delete(vpha1,idx_to_delete)
        eamp2 = np.delete(eamp2,idx_to_delete)
        epha2 = np.delete(epha2,idx_to_delete)
        namp2 = np.delete(namp2,idx_to_delete)
        npha2 = np.delete(npha2,idx_to_delete)
        vamp2 = np.delete(vamp2,idx_to_delete)
        vpha2 = np.delete(vpha2,idx_to_delete)
    # Search for and Remove any NaNs
    idx_to_delete = []
    for jj in range(0,len(eamp2)):
        # Current Station
        ceamp = eamp2[jj]
        # Just Test One Component for NaN (All Components Should be NaN If One Component Is)
        test = np.isnan(ceamp)
        if test:
            idx_to_delete.append(jj)
    # Gather All Indices Together into an Array, and Apply to Station Data
    idx_to_delete = np.asarray(idx_to_delete)
    if (len(idx_to_delete) > 0):
        sta = np.delete(sta,idx_to_delete)
        lat = np.delete(lat,idx_to_delete)
        lon = np.delete(lon,idx_to_delete)
        eamp1 = np.delete(eamp1,idx_to_delete)
        epha1 = np.delete(epha1,idx_to_delete)
        namp1 = np.delete(namp1,idx_to_delete)
        npha1 = np.delete(npha1,idx_to_delete)
        vamp1 = np.delete(vamp1,idx_to_delete)
        vpha1 = np.delete(vpha1,idx_to_delete)
        eamp2 = np.delete(eamp2,idx_to_delete)
        epha2 = np.delete(epha2,idx_to_delete)
        namp2 = np.delete(namp2,idx_to_delete)
        npha2 = np.delete(npha2,idx_to_delete)
        vamp2 = np.delete(vamp2,idx_to_delete)
        vpha2 = np.delete(vpha2,idx_to_delete)

    # Convert Amp+Phase to Real+Imag
    N1Re = np.multiply(namp1,np.cos(np.multiply(npha1,(pi/180.))))
    N1Im = np.multiply(namp1,np.sin(np.multiply(npha1,(pi/180.))))
    N2Re = np.multiply(namp2,np.cos(np.multiply(npha2,(pi/180.))))
    N2Im = np.multiply(namp2,np.sin(np.multiply(npha2,(pi/180.))))
    E1Re = np.multiply(eamp1,np.cos(np.multiply(epha1,(pi/180.))))
    E1Im = np.multiply(eamp1,np.sin(np.multiply(epha1,(pi/180.))))
    E2Re = np.multiply(eamp2,np.cos(np.multiply(epha2,(pi/180.))))
    E2Im = np.multiply(eamp2,np.sin(np.multiply(epha2,(pi/180.))))
    V1Re = np.multiply(vamp1,np.cos(np.multiply(vpha1,(pi/180.))))
    V1Im = np.multiply(vamp1,np.sin(np.multiply(vpha1,(pi/180.))))
    V2Re = np.multiply(vamp2,np.cos(np.multiply(vpha2,(pi/180.))))
    V2Im = np.multiply(vamp2,np.sin(np.multiply(vpha2,(pi/180.))))
 
    # Compute Differences
    NresRe = N1Re - N2Re
    NresIm = N1Im - N2Im
    EresRe = E1Re - E2Re
    EresIm = E1Im - E2Im
    VresRe = V1Re - V2Re
    VresIm = V1Im - V2Im

    # Optionally Remove the Common Mode
    if (rmCMode == True):

        # Optionally Exclude Stations from the Common-Mode Calculation
        # NOTE: An Earlier Version Had Not Been Excluding Stations from the Calculation Properly (Essentially, no Stations were Excluded)
        if stations_to_exclude:
            idxlst = []
            for ii in range(0,len(stations_to_exclude)):
                mysta = np.where(sta == stations_to_exclude[ii]); mysta = mysta[0]
                idxlst.extend(mysta)
            cmNresRe = np.delete(NresRe,idxlst); cmNresIm = np.delete(NresIm,idxlst)
            cmEresRe = np.delete(EresRe,idxlst); cmEresIm = np.delete(EresIm,idxlst)
            cmVresRe = np.delete(VresRe,idxlst); cmVresIm = np.delete(VresIm,idxlst)
            print('Stations Removed from Common-Mode Calculation: %s | Total Stations: %d | Stations Used for Common-Mode: %d' %(sta[idxlst], len(sta), len(cmVresRe)))
        else:
            cmNresRe = NresRe.copy(); cmNresIm = NresIm.copy()
            cmEresRe = EresRe.copy(); cmEresIm = EresIm.copy()
            cmVresRe = VresRe.copy(); cmVresIm = VresIm.copy()
            print('No Stations Removed from Common-Mode Calculation | Total Stations: %d | Stations Used for Common-Mode: %d' %(len(sta), len(cmNresRe)))

        # Compute Mean Values 
        meanNresRe = np.mean(cmNresRe)
        meanNresIm = np.mean(cmNresIm)
        meanEresRe = np.mean(cmEresRe)
        meanEresIm = np.mean(cmEresIm)
        meanVresRe = np.mean(cmVresRe)
        meanVresIm = np.mean(cmVresIm)

        # Determine the Mean Amplitudes and Phases (Later Appended to Station List as Common-Mode PME)
        meanNresAmp = np.sqrt(np.square(meanNresRe) + np.square(meanNresIm))
        meanNresPha = np.multiply(np.arctan2(meanNresIm,meanNresRe),(180./pi))
        meanEresAmp = np.sqrt(np.square(meanEresRe) + np.square(meanEresIm))
        meanEresPha = np.multiply(np.arctan2(meanEresIm,meanEresRe),(180./pi))
        meanVresAmp = np.sqrt(np.square(meanVresRe) + np.square(meanVresIm))
        meanVresPha = np.multiply(np.arctan2(meanVresIm,meanVresRe),(180./pi))
 
        # Remove the Mean Values
        NresRe = NresRe - meanNresRe
        NresIm = NresIm - meanNresIm
        EresRe = EresRe - meanEresRe
        EresIm = EresIm - meanEresIm
        VresRe = VresRe - meanVresRe
        VresIm = VresIm - meanVresIm

    # Compute Residual Amplitude and Phase
    NresAmp = np.sqrt(np.square(NresRe) + np.square(NresIm))
    NresPha = np.multiply(np.arctan2(NresIm,NresRe),(180./pi))
    EresAmp = np.sqrt(np.square(EresRe) + np.square(EresIm))
    EresPha = np.multiply(np.arctan2(EresIm,EresRe),(180./pi))
    VresAmp = np.sqrt(np.square(VresRe) + np.square(VresIm))
    VresPha = np.multiply(np.arctan2(VresIm,VresRe),(180./pi))

    # If Common-Mode Was Removed, Tack On the Common-Mode Ellipse to the End of the Station Arrays
    if (rmCMode == True):
        EresAmp = EresAmp.tolist(); EresAmp.append(meanEresAmp); EresAmp = np.asarray(EresAmp)
        EresPha = EresPha.tolist(); EresPha.append(meanEresPha); EresPha = np.asarray(EresPha)
        NresAmp = NresAmp.tolist(); NresAmp.append(meanNresAmp); NresAmp = np.asarray(NresAmp)
        NresPha = NresPha.tolist(); NresPha.append(meanNresPha); NresPha = np.asarray(NresPha)
        VresAmp = VresAmp.tolist(); VresAmp.append(meanVresAmp); VresAmp = np.asarray(VresAmp)
        VresPha = VresPha.tolist(); VresPha.append(meanVresPha); VresPha = np.asarray(VresPha)
        sta = sta.tolist(); sta.append('cMd*'); np.asarray(sta)
        lat = lat.tolist(); lat.append(45.0); np.asarray(lat)
        lon = lon.tolist(); lon.append(-45.0); np.asarray(lon)

    # Perform the Conversion
    smmjr,smmnr,theta = env2pme.main(EresAmp,EresPha,NresAmp,NresPha)

    # Search for and Remove any NaNs
    idx_to_delete = []
    for jj in range(0,len(EresAmp)):
        # Current Station
        cEres = EresAmp[jj]
        # Just Test One Component for NaN (All Components Should be NaN If One Component Is)
        test = np.isnan(cEres)
        if test:
            idx_to_delete.append(jj)
    # Gather All Indices Together into an Array, and Apply to Station Data
    idx_to_delete = np.asarray(idx_to_delete)
    if (len(idx_to_delete) > 0):
        sta = np.delete(sta,idx_to_delete)
        lat = np.delete(lat,idx_to_delete)
        lon = np.delete(lon,idx_to_delete)
        theta = np.delete(theta,idx_to_delete)
        smmjr = np.delete(smmjr,idx_to_delete)
        smmnr = np.delete(smmnr,idx_to_delete)
        EresAmp = np.delete(EresAmp,idx_to_delete)
        EresPha = np.delete(EresPha,idx_to_delete)
        NresAmp = np.delete(NresAmp,idx_to_delete)
        NresPha = np.delete(NresPha,idx_to_delete)
        VresAmp = np.delete(VresAmp,idx_to_delete)
        VresPha = np.delete(VresPha,idx_to_delete)

    # Search for and Remove Bad Stations
    idx_to_delete = []
    for jj in range(0,len(EresAmp)):
        # Current Station
        cSmmjr = smmjr[jj]
        # Just Test One Component for NaN (All Components Should be NaN If One Component Is)
        if (cSmmjr > 100.): # Semi-Major Axis of Residual is Greater Than XX mm 
            idx_to_delete.append(jj)
    # Gather All Indices Together into an Array, and Apply to Station Data
    idx_to_delete = np.asarray(idx_to_delete)
    if (len(idx_to_delete) > 0):
        sta = np.delete(sta,idx_to_delete)
        lat = np.delete(lat,idx_to_delete)
        lon = np.delete(lon,idx_to_delete)
        theta = np.delete(theta,idx_to_delete)
        smmjr = np.delete(smmjr,idx_to_delete)
        smmnr = np.delete(smmnr,idx_to_delete)
        EresAmp = np.delete(EresAmp,idx_to_delete)
        EresPha = np.delete(EresPha,idx_to_delete)
        NresAmp = np.delete(NresAmp,idx_to_delete)
        NresPha = np.delete(NresPha,idx_to_delete)
        VresAmp = np.delete(VresAmp,idx_to_delete)
        VresPha = np.delete(VresPha,idx_to_delete)

    # Prepare Output Files
    if (rmCMode == True):
        pme_file = (myoutfile[0:-4] + "_rmCMode.txt")
    else:
        pme_file = myoutfile
    pme_head = ("./pme_head.txt")
    pme_body = ("./pme_body.txt")

    # Prepare Data for Output (as Structured Array)
    all_pme_data = np.array(list(zip(sta,lat,lon,theta,smmjr,smmnr,EresAmp,EresPha,NresAmp,NresPha,VresAmp,VresPha)), \
        dtype=[('sta','U8'), \
        ('lat',float),('lon',float),('theta',float),('smmjr',float),('smmnr',float),('EresAmp',float), \
        ('EresPha',float),('NresAmp',float),('NresPha',float),('VresAmp',float),('VresPha',float)])

    # Write Header Info to File
    hf = open(pme_head,'w')
    pme_str = 'Station  Lat(+N,deg)  Lon(+E,deg)  Direction(deg)  Semi-Major(mm)  Semi-Minor(mm)  E-Amp(mm)  E-Pha(deg)  N-Amp(mm)  N-Pha(deg)  V-Amp(mm)  V-Pha(deg) \n'
    hf.write(pme_str)
    hf.close()

    # Write PME Results to File
    #f_handle = open(pme_body,'w')
    np.savetxt(pme_body,all_pme_data,fmt=["%s"]+["%.7f",]*11,delimiter="        ")
    #f_handle.close()

    # Combine Header and Body Files
    filenames_pme = [pme_head, pme_body]
    with open(pme_file,'w') as outfile:
        for fname in filenames_pme:
            with open(fname) as infile:
                outfile.write(infile.read())
 
    # Remove Header and Body Files
    os.remove(pme_head)
    os.remove(pme_body)

