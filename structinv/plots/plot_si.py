#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO PLOT RESULTS FROM THE INVERSION FOR STRUCTURE
# 
# Copyright (c) 2023-2024: HILARY R. MARTENS
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

# IMPORT PYTHON MODULES
import numpy as np
import matplotlib.pyplot as plt
from LOADGF.LN import prepare_planet_model
from scipy import interpolate

# Provide Path Names to Planet Models (May be Multiple, Provided in List Format)
earthmods = [("../../input/Planet_Models/STW105.txt"),\
    ("../output/Inversion/iv_newModel_pme_OceanOnly_M2_cm_convgf_GOT410c_commonMesh_STW105_TikReg-second-1.txt"),\
    ("../../input/Planet_Models/PREM.txt")]
labels = ["DataModel-STW105",\
    "Recovered-Alpha1",\
    "StartingModel-PREM"]
colors = ['k','b','k']
weights = [2,2,1]
styles = ['-','-',':']

# Universal Gravitational Constant
G = 6.672E-11

# Maximum Depth for Plot (km)
max_depth = 500.

# Figure Name
figname = ("Recovered_Model.pdf")

#### BEGIN CODE

# Create Folder
if not (os.path.isdir("./output/")):
    os.makedirs("./output/")
outdir = "./output/"

# Initialize Figure 1
fig = plt.figure()
# Top row
ax1 = plt.subplot2grid((3, 6), (0, 0), colspan=3)
ax2 = plt.subplot2grid((3, 6), (0, 3), colspan=3)
# Middle row
ax3 = plt.subplot2grid((3, 6), (1, 0), colspan=2)
ax4 = plt.subplot2grid((3, 6), (1, 2), colspan=2)
ax5 = plt.subplot2grid((3, 6), (1, 4), colspan=2)
# Bottom row
ax6 = plt.subplot2grid((3, 6), (2, 0), colspan=2)
ax7 = plt.subplot2grid((3, 6), (2, 2), colspan=2)
ax8 = plt.subplot2grid((3, 6), (2, 4), colspan=2)
# Initialize Arrays
all_ir = []
all_ivp = []
all_ivs = []
all_imu = []
all_iK = []
all_irho = []
all_ilmda = []
# Loop Through All Earth Models
for ii in range(0,len(earthmods)):

    # Current Model
    myfile = earthmods[ii]
    mycolor = colors[ii]
    mylabel = labels[ii]
    myweight = weights[ii]
    mystyle = styles[ii]

    # Prepare Earth Model
    ir,imu,iK,ilmda,irho,ig,tck_lnd,tck_mnd,tck_rnd,tck_gnd,s,lnd,mnd,rnd,gnd,s_min,small,\
        earth_radius,earth_mass,sic,soc,adim,gsdim,pi,piG,L_sc,R_sc,T_sc = prepare_planet_model.main(myfile)

    # Compute P- and S-wave Velocities
    ivp = np.sqrt(np.divide(iK + (4./3.)*imu, irho))
    ivs = np.sqrt(np.divide(imu, irho))

    # Convert Radial Distances to km
    ir = ir/1000.

    # Set Main Radial Vector (needed later for interpolation)
    if (ii == 0):
        main_ir = ir.copy()
    
    # Convert Other Parameters to Meaninful Units
    ivp = ivp/1000.
    ivs = ivs/1000.

    # Convert Zero Values to Slightly Non-Zero Values in Preparation for Taking the Logarithms Later (Exactly Zero Values are Unphysical Anyway)
    zero_idx = np.where(ivs == 0); zero_idx = zero_idx[0]
    ivs[zero_idx] = 1E-17
    zero_idx = np.where(imu == 0); zero_idx = zero_idx[0]
    imu[zero_idx] = 1E-17
    zero_idx = np.where(ilmda == 0); zero_idx = zero_idx[0]
    ilmda[zero_idx] = 1E-17

    # Convert Material Parameters to Log Space
    irho = np.log10(irho)
    imu = np.log10(imu)
    iK = np.log10(iK)

    # Interpolate to "Main" Radial Vector
    if (ii > 0):
        # Specify order of interpolation
        kx = 1 #(1 = linear)
        # Interpolate
        tck_mu = interpolate.splrep(ir,imu,k=kx)
        imu = interpolate.splev(main_ir,tck_mu,der=0)
        tck_K = interpolate.splrep(ir,iK,k=kx)
        iK = interpolate.splev(main_ir,tck_K,der=0)
        tck_lmda = interpolate.splrep(ir,ilmda,k=kx)
        ilmda = interpolate.splev(main_ir,tck_lmda,der=0)
        tck_rho = interpolate.splrep(ir,irho,k=kx)
        irho = interpolate.splev(main_ir,tck_rho,der=0)
        tck_vp = interpolate.splrep(ir,ivp,k=kx)
        ivp = interpolate.splev(main_ir,tck_vp,der=0)
        tck_vs = interpolate.splrep(ir,ivs,k=kx)
        ivs = interpolate.splev(main_ir,tck_vs,der=0)
        ir = main_ir.copy()
        # Convert Lists Back to Arrays
        imu = np.asarray(imu)
        iK = np.asarray(iK)
        irho = np.asarray(irho)
        ilmda = np.asarray(ilmda)
        ivp = np.asarray(ivp)
        ivs = np.asarray(ivs)

    # Append Info to Arrays
    all_ir.append(ir)
    all_ivp.append(ivp)
    all_ivs.append(ivs)
    all_irho.append(irho)
    all_imu.append(imu)
    all_iK.append(iK)
    all_ilmda.append(ilmda)

    # Compute Depth
    depth = (earth_radius/1000.) - ir

    # Add to Axes
    ax1.plot(ivp,depth,color=mycolor,label=mylabel,linewidth=myweight,linestyle=mystyle)
    ax1.set_title(r'$\mathrm{V_P}$ [km/s]', fontsize='xx-small')
    ax1.set_ylim(0.,max_depth)
    ax1.set_xlim(5,12)
    ax1.invert_yaxis()
    ax1.set_ylabel('Depth [km] ', fontsize='xx-small')
    ax1.tick_params(labelsize=5)
    ax1.legend(loc='lower left',fontsize='xx-small')
    ax1.grid(True)
    ax1.text(11.1,180,'A',horizontalalignment='left',size='small')
    ax2.plot(ivs,depth,color=mycolor,linewidth=myweight,linestyle=mystyle)
    ax2.set_title(r'$\mathrm{V_S}$ [km/s]', fontsize='xx-small')
    ax2.set_ylim(0.,max_depth)
    ax2.set_xlim(3,7)
    ax2.invert_yaxis()
    ax2.tick_params(labelsize=5)
    ax2.grid(True)
    ax2.text(6.55,180,'B',horizontalalignment='left',size='small')
    ax3.plot(imu,depth,color=mycolor,linewidth=myweight,linestyle=mystyle)
    ax3.set_title(r'log$_{10}\,\mu$ [log$_{10}$(Pa)]', fontsize='xx-small')
    ax3.set_ylim(0.,max_depth)
    ax3.set_xlim(10.3,11.3)
    ax3.invert_yaxis()
    ax3.set_ylabel('Depth [km] ',fontsize='xx-small')
    ax3.tick_params(labelsize=5)
    ax3.grid(True) 
    ax3.text(11.1,180,'C',horizontalalignment='left',size='small')
    ax4.plot(iK,depth,color=mycolor,linewidth=myweight,linestyle=mystyle)
    ax4.set_title(r'log$_{10}\,\kappa$ [log$_{10}$(Pa)]',fontsize='xx-small')
    ax4.set_ylim(0.,max_depth)
    ax4.set_xlim(10.6,11.6)
    ax4.invert_yaxis()
    ax4.tick_params(labelsize=5)
    ax4.grid(True)
    ax4.text(11.42,180,'D',horizontalalignment='left',size='small')
    ax5.plot(irho,depth,color=mycolor,linewidth=myweight,linestyle=mystyle)
    ax5.set_title(r'log$_{10}\,\rho$ [log$_{10}$(kg/m$^3$)]', fontsize='xx-small')
    ax5.set_ylim(0.,max_depth)
    ax5.set_xlim(3.4,3.7)
    ax5.invert_yaxis()
    ax5.tick_params(labelsize=5)
    ax5.grid(True)
    ax5.text(3.66,180,'E',horizontalalignment='left',size='small')
# Convert Lists to Numpy Arrays
all_ir = np.asarray(all_ir)
all_ivp = np.asarray(all_ivp)
all_ivs = np.asarray(all_ivs)
all_irho = np.asarray(all_irho)
all_imu = np.asarray(all_imu)
all_iK = np.asarray(all_iK)
all_ilmda = np.asarray(all_ilmda)
# Now Determine the Maximum % Difference Between All Models at Each Depth Level
min_all_ivp = np.amin(all_ivp,axis=0)
max_all_ivp = np.amax(all_ivp,axis=0)
mean_all_ivp = np.mean(all_ivp,axis=0)
diff_all_ivp = max_all_ivp - min_all_ivp
perc_diff_ivp = ((max_all_ivp - min_all_ivp)/min_all_ivp)*100.
min_all_ivs = np.amin(all_ivs,axis=0)
max_all_ivs = np.amax(all_ivs,axis=0)
mean_all_ivs = np.mean(all_ivs,axis=0)
ivs_nonzero = np.where(min_all_ivs > 0.00000001)
perc_diff_ivs = np.zeros((len(mean_all_ivs),))
perc_diff_ivs[ivs_nonzero] = ((max_all_ivs[ivs_nonzero] - min_all_ivs[ivs_nonzero])/min_all_ivs[ivs_nonzero])*100.
min_all_irho = np.amin(all_irho,axis=0)
max_all_irho = np.amax(all_irho,axis=0)
mean_all_irho = np.mean(all_irho,axis=0)
perc_diff_irho = ((max_all_irho - min_all_irho)/min_all_irho)*100.
min_all_imu = np.amin(all_imu,axis=0)
max_all_imu = np.amax(all_imu,axis=0)
mean_all_imu = np.mean(all_imu,axis=0)
imu_nonzero = np.where(min_all_imu > 0.00000001)
perc_diff_imu = np.zeros((len(mean_all_imu),))
perc_diff_imu[imu_nonzero] = ((max_all_imu[imu_nonzero] - min_all_imu[imu_nonzero])/min_all_imu[imu_nonzero])*100.
min_all_iK = np.amin(all_iK,axis=0)
max_all_iK = np.amax(all_iK,axis=0)
mean_all_iK = np.mean(all_iK,axis=0)
perc_diff_iK = ((max_all_iK - min_all_iK)/min_all_iK)*100.
min_all_ilmda = np.amin(all_ilmda,axis=0)
max_all_ilmda = np.amax(all_ilmda,axis=0)
mean_all_ilmda = np.mean(all_ilmda,axis=0)
ilmda_nonzero = np.where(min_all_ilmda > 0.00000001)
perc_diff_ilmda = np.zeros((len(mean_all_ilmda),))
perc_diff_ilmda[ilmda_nonzero] = ((max_all_ilmda[ilmda_nonzero] - min_all_ilmda[ilmda_nonzero])/min_all_ilmda[ilmda_nonzero])*100.
# Plot Remaining Axes
ax6.plot(perc_diff_imu,depth,'k',linewidth=2)
ax6.set_title(r'Max $\Delta$ log$_{10}\,\mu$ [%]', fontsize='xx-small')
ax6.set_ylim(0.,max_depth)
ax6.set_xlim(0,4)
ax6.invert_yaxis()
ax6.tick_params(labelsize=5)
ax6.set_ylabel('Depth [km] ',fontsize='xx-small')
ax6.grid()
ax6.text(3.55,180,'F',horizontalalignment='left',size='small')
ax7.plot(perc_diff_iK,depth,'k',linewidth=2)
ax7.set_title(r'Max $\Delta$ log$_{10}\,\kappa$ [%]', fontsize='xx-small')
ax7.set_ylim(0.,max_depth)
ax7.set_xlim(0,4)
ax7.invert_yaxis()
ax7.tick_params(labelsize=5)
ax7.grid()
ax7.text(3.55,180,'G',horizontalalignment='left',size='small')
ax8.plot(perc_diff_irho,depth,'k',linewidth=2)
ax8.set_title(r'Max $\Delta$ log$_{10}\,\rho$ [%]',fontsize='xx-small')
ax8.set_ylim(0.,max_depth)
ax8.set_xlim(0,4)
ax8.invert_yaxis()
ax8.tick_params(labelsize=5)
ax8.grid()
ax8.text(3.55,180,'H',horizontalalignment='left',size='small')
# Save and Plot the Figure
plt.tight_layout()
plt.savefig((outdir+figname),orientation='portrait',format='pdf')
plt.show()


