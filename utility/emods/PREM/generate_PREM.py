# *********************************************************************
# PROGRAM TO GENERATE THE PREM MODEL FROM POLYNOMIAL FUNCTIONS
# LITERATURE: Dziewonski & Anderson (1981)
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

from __future__ import print_function
import numpy as np
import sys
import os

def main(interval_cores,interval_lowerMantle,interval_upperMantle,interval_crust,outfile):

    # Earth Radius
    radius = 6371.0

    # Outfile - Vp,Vs only
    outfilevpvs = os.path.splitext(outfile)[0]+'_VpVs.txt'
    print(outfilevpvs)

    # Initialize Lists
    rho = []
    vp  = []
    vs  = []
    Qmu = []
    QK  = []
    rdist = []

    # Inner Core
    ic = np.linspace(0.,1221.5,num=(1221.5/interval_cores)+1.,endpoint=True)
    icnorm = np.divide(ic,radius)
    icrho = 13.0885-8.8381*np.square(icnorm)
    icvp = 11.2622-6.3640*np.square(icnorm)
    icvs = 3.6678-4.4475*np.square(icnorm)
    icQmu = np.ones((len(ic),))*84.6
    icQK = np.ones((len(ic),))*1327.7
    rho.extend(icrho)
    vp.extend(icvp)
    vs.extend(icvs)
    Qmu.extend(icQmu)
    QK.extend(icQK)
    rdist.extend(ic) 

    # Outer Core
    oc = np.linspace(1221.5,3480.0,num=((3480.0-1221.5)/interval_cores)+1.,endpoint=True)
    ocnorm = np.divide(oc,radius)
    ocrho = 12.5815-1.2638*ocnorm-3.6426*np.square(ocnorm)-5.5281*np.power(ocnorm,3.)
    ocvp = 11.0487-4.0362*ocnorm+4.8023*np.square(ocnorm)-13.5732*np.power(ocnorm,3.)
    ocvs = np.zeros((len(ocnorm),))
    ocQmu = np.ones((len(oc),))*np.inf
    ocQK = np.ones((len(oc),))*57823
    rho.extend(ocrho)
    vp.extend(ocvp)
    vs.extend(ocvs)
    Qmu.extend(ocQmu)
    QK.extend(ocQK)
    rdist.extend(oc)
    
    # Lower Mantle
    lm = np.linspace(3480.0,3630.0,num=((3630.0-3480.0)/interval_lowerMantle)+1.,endpoint=True)
    lmnorm = np.divide(lm,radius)
    lmrho = 7.9565-6.4761*lmnorm+5.5283*np.square(lmnorm)-3.0807*np.power(lmnorm,3.)
    lmvp = 15.3891-5.3181*lmnorm+5.5242*np.square(lmnorm)-2.5514*np.power(lmnorm,3.)
    lmvs = 6.9254+1.4672*lmnorm-2.0834*np.square(lmnorm)+0.9783*np.power(lmnorm,3.)
    lmQmu = np.ones((len(lm),))*312
    lmQK = np.ones((len(lm),))*57823
    rho.extend(lmrho)
    vp.extend(lmvp)
    vs.extend(lmvs)
    Qmu.extend(lmQmu)
    QK.extend(lmQK)
    rdist.extend(lm)

    # Lower Mantle 2a
#    lm = np.linspace(3630.0,5200.0,num=((5200.0-3630.0)/interval_lowerMantle)+1.,endpoint=True)
    lm = np.linspace(3630.0,5600.0,num=((5600.0-3630.0)/interval_lowerMantle)+1.,endpoint=True)
    lmnorm = np.divide(lm,radius)
    lmrho = 7.9565-6.4761*lmnorm+5.5283*np.square(lmnorm)-3.0807*np.power(lmnorm,3.)
    lmvp = 24.9520-40.4673*lmnorm+51.4832*np.square(lmnorm)-26.6419*np.power(lmnorm,3.)
    lmvs = 11.1671-13.7818*lmnorm+17.4575*np.square(lmnorm)-9.2777*np.power(lmnorm,3.)
    lmQmu = np.ones((len(lm),))*312
    lmQK = np.ones((len(lm),))*57823
    rho.extend(lmrho)
    vp.extend(lmvp)
    vs.extend(lmvs)
    Qmu.extend(lmQmu)
    QK.extend(lmQK)
    rdist.extend(lm)

    # Lower Mantle 2b
#    lm = np.linspace(5200.0,5600.0,num=((5600.0-5200.0)/interval_upperMantle)+1.,endpoint=True)
#    lmnorm = np.divide(lm,radius)
#    lmrho = 7.9565-6.4761*lmnorm+5.5283*np.square(lmnorm)-3.0807*np.power(lmnorm,3.)
#    lmvp = 24.9520-40.4673*lmnorm+51.4832*np.square(lmnorm)-26.6419*np.power(lmnorm,3.)
#    lmvs = 11.1671-13.7818*lmnorm+17.4575*np.square(lmnorm)-9.2777*np.power(lmnorm,3.)
#    lmQmu = np.ones((len(lm),))*312
#    lmQK = np.ones((len(lm),))*57823
#    rho.extend(lmrho)
#    vp.extend(lmvp)
#    vs.extend(lmvs)
#    Qmu.extend(lmQmu)
#    QK.extend(lmQK)
#    rdist.extend(lm)

    # Lower Mantle 3
    lm = np.linspace(5600.0,5701.0,num=((5701.0-5600.0)/interval_upperMantle)+1.,endpoint=True)
    lmnorm = np.divide(lm,radius)
    lmrho = 7.9565-6.4761*lmnorm+5.5283*np.square(lmnorm)-3.0807*np.power(lmnorm,3.)
    lmvp = 29.2766-23.6027*lmnorm+5.5242*np.square(lmnorm)-2.5514*np.power(lmnorm,3.)
    lmvs = 22.3459-17.2473*lmnorm-2.0834*np.square(lmnorm)+0.9783*np.power(lmnorm,3.)
    lmQmu = np.ones((len(lm),))*312
    lmQK = np.ones((len(lm),))*57823
    rho.extend(lmrho)
    vp.extend(lmvp)
    vs.extend(lmvs)
    Qmu.extend(lmQmu)
    QK.extend(lmQK)
    rdist.extend(lm)

    # Transition Zone 1
    tz = np.linspace(5701.0,5771.0,num=((5771.0-5701.0)/interval_upperMantle)+1.,endpoint=True)
    tznorm = np.divide(tz,radius)
    tzrho = 5.3197-1.4836*tznorm
    tzvp = 19.0957-9.8672*tznorm
    tzvs = 9.9839-4.9324*tznorm
    tzQmu = np.ones((len(tz),))*143
    tzQK = np.ones((len(tz),))*57823
    rho.extend(tzrho)
    vp.extend(tzvp)
    vs.extend(tzvs)
    Qmu.extend(tzQmu)
    QK.extend(tzQK)
    rdist.extend(tz)

    # Transition Zone 2
    tz = np.linspace(5771.0,5971.0,num=((5971.0-5771.0)/interval_upperMantle)+1.,endpoint=True)
    tznorm = np.divide(tz,radius)
    tzrho = 11.2494-8.0298*tznorm
    tzvp = 39.7027-32.6166*tznorm
    tzvs = 22.3512-18.5856*tznorm
    tzQmu = np.ones((len(tz),))*143
    tzQK = np.ones((len(tz),))*57823
    rho.extend(tzrho)
    vp.extend(tzvp)
    vs.extend(tzvs)
    Qmu.extend(tzQmu)
    QK.extend(tzQK)
    rdist.extend(tz)

    # Transition Zone 3
    tz = np.linspace(5971.0,6151.0,num=((6151.0-5971.0)/interval_upperMantle)+1.,endpoint=True)
    tznorm = np.divide(tz,radius)
    tzrho = 7.1089-3.8045*tznorm
    tzvp = 20.3926-12.2569*tznorm
    tzvs = 8.9496-4.4597*tznorm
    tzQmu = np.ones((len(tz),))*143
    tzQK = np.ones((len(tz),))*57823
    rho.extend(tzrho)
    vp.extend(tzvp)
    vs.extend(tzvs)
    Qmu.extend(tzQmu)
    QK.extend(tzQK)
    rdist.extend(tz)

    # LVZ
    lv = np.linspace(6151.0,6291.0,num=((6291.0-6151.0)/interval_upperMantle)+1.,endpoint=True)
    lvnorm = np.divide(lv,radius)
    lvrho = 2.6910+0.6924*lvnorm
    # Effective Isotropic Velocities
    lvvp = 4.1875+3.9382*lvnorm
    lvvs = 2.1519+2.3481*lvnorm
#    lvvp = 0.8317+7.2180*lvnorm
#    lvvs = 5.8582-1.4678*lvnorm
    lvQmu = np.ones((len(lv),))*80
    lvQK = np.ones((len(lv),))*57823
    rho.extend(lvrho)
    vp.extend(lvvp)
    vs.extend(lvvs)
    Qmu.extend(lvQmu)
    QK.extend(lvQK)
    rdist.extend(lv)

    # LID
    ld = np.linspace(6291.0,6346.6,num=((6346.6-6291.0)/interval_upperMantle)+1.,endpoint=True)
    ldnorm = np.divide(ld,radius)
    ldrho = 2.6910+0.6924*ldnorm
    # Effective Isotropic Velocities
    ldvp = 4.1875+3.9382*ldnorm
    ldvs = 2.1519+2.3481*ldnorm
#    ldvp = 0.8317+7.2180*ldnorm
#    ldvs = 5.8582-1.4678*ldnorm
    ldQmu = np.ones((len(ld),))*600
    ldQK = np.ones((len(ld),))*57823
    rho.extend(ldrho)
    vp.extend(ldvp)
    vs.extend(ldvs)
    Qmu.extend(ldQmu)
    QK.extend(ldQK)
    rdist.extend(ld)

    # Crust 1
    cr = np.linspace(6346.6,6356.0,num=((6356.0-6346.6)/interval_crust)+1.,endpoint=True)
    crnorm = np.divide(cr,radius)
    crrho = np.ones((len(cr),))*2.900
    crvp = np.ones((len(cr),))*6.800
    crvs = np.ones((len(cr),))*3.900
    crQmu = np.ones((len(cr),))*600
    crQK = np.ones((len(cr),))*57823
    rho.extend(crrho)
    vp.extend(crvp)
    vs.extend(crvs)
    Qmu.extend(crQmu)
    QK.extend(crQK)
    rdist.extend(cr)

    # Crust 2
    cr = np.linspace(6356.0,6368.0,num=((6368.0-6356.0)/interval_crust)+1.,endpoint=True)
    crnorm = np.divide(cr,radius)
    crrho = np.ones((len(cr),))*2.600
    crvp = np.ones((len(cr),))*5.800
    crvs = np.ones((len(cr),))*3.200
    crQmu = np.ones((len(cr),))*600
    crQK = np.ones((len(cr),))*57823
    rho.extend(crrho)
    vp.extend(crvp)
    vs.extend(crvs)
    Qmu.extend(crQmu)
    QK.extend(crQK)
    rdist.extend(cr)

    # Top Layer
    cr = np.linspace(6368.0,6371.0,num=((6371.0-6368.0)/interval_crust)+1.,endpoint=True)
    crnorm = np.divide(cr,radius)
    crrho = np.ones((len(cr),))*1.020
    crvp = np.ones((len(cr),))*1.450
    crvs = np.ones((len(cr),))*0.000
    crQmu = np.ones((len(cr),))*np.inf
    crQK = np.ones((len(cr),))*57823
    rho.extend(crrho)
    vp.extend(crvp)
    vs.extend(crvs)
    Qmu.extend(crQmu)
    QK.extend(crQK)
    rdist.extend(cr)

    # :: As in Guo et al. (2004), average the density of the ocean and upper-most crustal layers, conserving total mass
    # :: Keep the elastic moduli equivalent to the original upper-most crustal layer
    # Volume of ocean layer:
    Vocean = (4./3)*np.pi*(6371.**3) - (4./3)*np.pi*(6368.**3)
    # Volume of upper-most crustal layer:
    Vcrust = (4./3)*np.pi*(6368.**3) - (4./3)*np.pi*(6356.**3)
    # Total volume of ocean and upper-most crustal layers
    Vtotal = Vocean + Vcrust
    # Mass of ocean layer:
    Mocean = Vocean * 1.020
    # Mass of upper-most crustal layer:
    Mcrust = Vcrust * 2.600
    # Total mass of ocean and upper-most crustal layers:
    Mtotal = Mocean + Mcrust
    # Weighted-average density of new, combined layer -- conserving total mass
    new_rho = Mtotal / Vtotal
    # Compute mu from upper-most crustal layer:
    mu = (3200.**2)*2600.
    # Compute kappa from upper-most crustal layer:
    kappa = (5800.**2)*2600. - (4./3)*mu
    # Re-compute Vp and Vs with new layer density:
    new_vp = np.sqrt((kappa + (4./3)*mu)/(new_rho*1000.))
    new_vs = np.sqrt(mu/(new_rho*1000.))
    # Convert Vp and Vs to km/s
    new_vp /= 1000.
    new_vs /= 1000.
    # Replace old values with new values
    rdist = np.asarray(rdist)
    vp = np.asarray(vp)
    vs = np.asarray(vs)
    rho = np.asarray(rho)
    QK = np.asarray(QK)
    Qmu = np.asarray(Qmu)
    upper_layer = np.where((rdist >= 6356.) & (vp < 6.0)); upper_layer = upper_layer[0]
    rho[upper_layer] = new_rho
    vp[upper_layer] = new_vp
    vs[upper_layer] = new_vs

    # Print to File
    all_data = np.array(list(zip(rdist,vp,vs,rho,QK,Qmu)),dtype=[('rdist',float),('vp',float),\
        ('vs',float),('rho',float),('QK',float),('Qmu',float)])
    #f_handle = open(outfile,'w')
    np.savetxt(outfile,all_data,fmt=["%0.3f"]+["%0.7f",]*5,delimiter="   ")
    #f_handle.close()    

    # Print to File (rdist, vp, vs only)
    all_data = np.array(list(zip(rdist,vp,vs)),dtype=[('rdist',float),('vp',float),('vs',float)])
    #f_handle = open(outfilevpvs,'w')
    np.savetxt(outfilevpvs,all_data,fmt=["%.1f"]+["%0.3f",]*2,delimiter="   ")
    #f_handle.close()

    # Return Variables
    return rdist,vp,vs,rho,QK,Qmu


