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
import numpy as np
import math
import sys
from scipy import interpolate
from scipy.integrate import simps
import matplotlib.pyplot as plt
from LOADGF.LN import f_solid_n0
from LOADGF.LN import f_solid
from LOADGF.PL import okubo_saito_dI

def main(n,sint,Yload,Ypot,Yshr,Ystr,hload,nlload,nkload,hpot,nlpot,nkpot,hshr,nlshr,nkshr,hstr,nlstr,nkstr,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m,plot_figure):

    # Normalized Radius and Gravity at Surface
    a_norm = sint[-1]
    gs_norm = interpolate.splev(a_norm,tck_gnd,der=0)

    # Normalized Rho and g at 'sint' Radii
    rho_norm = interpolate.splev(sint,tck_rnd,der=0)
    g_norm   = interpolate.splev(sint,tck_gnd,der=0)

    # Special Case: n=0
    if (n == 0):
 
        # Convert 6-component Y vectors to 4-components (Remove NaNs)
        Yload1   = Yload[:,0]; Yload2 = Yload[:,1]; Yload5 = Yload[:,4]; Yload6 = Yload[:,5]
        Ypot1    = Ypot[:,0];  Ypot2  = Ypot[:,1];  Ypot5  = Ypot[:,4];  Ypot6  = Ypot[:,5]
        Yshr1    = Yshr[:,0];  Yshr2  = Yshr[:,1];  Yshr5  = Yshr[:,4];  Yshr6  = Yshr[:,5]
        Ystr1    = Ystr[:,0];  Ystr2  = Ystr[:,1];  Ystr5  = Ystr[:,4];  Ystr6  = Ystr[:,5]
        Yload_n0 = np.column_stack((Yload1,Yload2,Yload5,Yload6))
        Ypot_n0  = np.column_stack((Ypot1, Ypot2, Ypot5, Ypot6))
        Yshr_n0  = np.column_stack((Yshr1, Yshr2, Yshr5, Yshr6))
        Ystr_n0  = np.column_stack((Ystr1, Ystr2, Ystr5, Ystr6))

        # Compute dY/dr
        YP_load = Yload_n0.copy()
        YP_pot  = Ypot_n0.copy()
        YP_shr  = Yshr_n0.copy()
        YP_str  = Ystr_n0.copy()
        for yy in range(0,len(sint)):
            YP_load[yy,:] = f_solid_n0.main(sint[yy],Yload_n0[yy,:].T,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG)
            YP_pot[yy,:]  = f_solid_n0.main(sint[yy],Ypot_n0[yy,:].T,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG)
            YP_shr[yy,:]  = f_solid_n0.main(sint[yy],Yshr_n0[yy,:].T,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG)
            YP_str[yy,:]  = f_solid_n0.main(sint[yy],Ystr_n0[yy,:].T,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG)

        # Add Y3 and Y4 Back In As Zeros to Y and YP Arrays
        Y3 = np.zeros((len(sint),))
        Y4 = Y3.copy()
        Yload[:,2] = Y3.copy()
        Yload[:,3] = Y3.copy()
        Ypot[:,2] = Y3.copy()
        Ypot[:,3] = Y3.copy()
        Yshr[:,2] = Y3.copy()
        Yshr[:,3] = Y3.copy()
        Ystr[:,2] = Y3.copy()
        Ystr[:,3] = Y3.copy()
        YPload = np.column_stack((YP_load[:,0],YP_load[:,1],Y3,Y4,YP_load[:,2],YP_load[:,3]))
        YPpot = np.column_stack((YP_pot[:,0],YP_pot[:,1],Y3,Y4,YP_pot[:,2],YP_pot[:,3]))
        YPshr = np.column_stack((YP_shr[:,0],YP_shr[:,1],Y3,Y4,YP_shr[:,2],YP_shr[:,3]))
        YPstr = np.column_stack((YP_str[:,0],YP_str[:,1],Y3,Y4,YP_str[:,2],YP_str[:,3]))    

    # Special Case: n=1
    elif (n == 1):

        # Convert Y-Solutions to Okubo & Saito (1983) and Okubo & Endo (1986) Convention
        Yload = Yload / (a_norm * gs_norm)
        Ystr  = Ystr  / (a_norm * gs_norm)
        Ypot  = Ypot  / (a_norm * gs_norm)
        Yshr  = Yshr  / (a_norm * gs_norm)       
 
        # Add a Rigid Body Rotation (Merriam, 1985)
        #  Equivalent to Takeuchi & Saito (1972) Eq. 100 for n=1
        # STRESS
        alpha = -(Ystr[-1,4]/gs_norm) # Define Alpha Such That k_1 = 0, or Y5(a) = 0 (Okubo & Endo 1986, Sec. 3)
        Ystr[:,0] += (1.*alpha)
        Ystr[:,1] += 0.
        Ystr[:,2] += (1.*alpha)
        Ystr[:,3] += 0.
        Ystr[:,4] += g_norm*alpha
        Ystr[:,5] += -2.*np.divide(g_norm,sint)*alpha
        # LOAD
        alpha = -(Yload[-1,4]/gs_norm) + (1./gs_norm) # Define Alpha Such That k_1 = 0, or Y5(a) = 1 (Okubo & Endo 1986, Sec. 3)
        Yload[:,0] += (1.*alpha)
        Yload[:,1] += 0.
        Yload[:,2] += (1.*alpha)
        Yload[:,3] += 0.
        Yload[:,4] += g_norm*alpha
        Yload[:,5] += -2.*np.divide(g_norm,sint)*alpha 

        # Dimensional Constants
        pi = np.pi
        G = 6.672E-11
        R_sc = 5500.                         # density
        T_sc = 1./(math.sqrt(R_sc*pi*G))     # time
        L_sc = 6371000.
        P_norm = ((L_sc**2.)*(T_sc**(-2.)))/1E6
        R_norm = L_sc/1000.
        G_norm = (L_sc*(T_sc**(-2)))

        # Compute Love Numbers
        hload = Yload[-1,0]*gs_norm
        nlload = Yload[-1,2]*gs_norm
        nkload = Yload[-1,4] - 1.
        #print(':: n=1 : hload, nlload, nkload : ', hload, nlload, nkload)
        hstr = Ystr[-1,0]*gs_norm
        nlstr = Ystr[-1,2]*gs_norm
        nkstr = Ystr[-1,4]
        #print(':: n=1 : hstr, nlstr, nkstr : ', hstr, nlstr, nkstr)

        # Plot Y-Solutions (Reproduce Okubo & Endo 1986, Fig. 1)
        if (plot_figure == True):
            plt.figure()
            plt.subplot(2,3,1)
            plt.plot(Yload[:,0]*(R_norm/P_norm),sint)
            plt.xlim([-40.,100.])
            plt.title('Y_1 Load', fontsize='xx-small')
            plt.grid(True)
            plt.tick_params(labelsize='xx-small')
            plt.subplot(2,3,2)
            plt.plot(Yload[:,2]*(R_norm/P_norm),sint)
            plt.xlim([-40.,100.])
            plt.title('Y_3 Load', fontsize='xx-small')
            plt.grid(True)
            plt.tick_params(labelsize='xx-small')
            plt.subplot(2,3,3)
            plt.plot(Yload[:,4]*(P_norm/P_norm),sint)
            plt.title('Y_5 Load', fontsize='xx-small')
            plt.grid(True)
            plt.tick_params(labelsize='xx-small')
            plt.xlim([0,1.])
            plt.subplot(2,3,4)
            plt.plot(Ystr[:,0]*(R_norm/P_norm),sint)
            plt.xlim([-40.,100.])
            plt.title('Y_1 Stress', fontsize='xx-small')
            plt.grid(True)
            plt.tick_params(labelsize='xx-small')
            plt.subplot(2,3,5)
            plt.plot(Ystr[:,2]*(R_norm/P_norm),sint)
            plt.xlim([-40.,100.])
            plt.title('Y_3 Stress', fontsize='xx-small')
            plt.grid(True)
            plt.tick_params(labelsize='xx-small')
            plt.subplot(2,3,6)
            plt.plot(Ystr[:,4]*(P_norm/P_norm),sint)
            plt.xlim([0,0.1])
            plt.title('Y_5 Stress', fontsize='xx-small')
            plt.tick_params(labelsize='xx-small')
            plt.grid(True)
            print(':: Reproducing Figure 1 from Okubo & Endo (1986). Returning Figure to ../output/Love_Numbers/Partials/ Directory. [ln_partials.py]')
            plt.savefig('../output/Love_Numbers/Partials/Okubo_Endo_Fig1.eps',orientation='landscape',format='eps')

        # Compute dY/dr
        YPload = Yload.copy()
        YPpot  = Ypot.copy()
        YPshr  = Yshr.copy()
        YPstr  = Ystr.copy()
        for yy in range(0,len(sint)):
            YPload[yy,:] = f_solid.main(sint[yy],Yload[yy,:].T,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)
            YPpot[yy,:]  = f_solid.main(sint[yy],Ypot[yy,:].T,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)
            YPshr[yy,:]  = f_solid.main(sint[yy],Yshr[yy,:].T,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)
            YPstr[yy,:]  = f_solid.main(sint[yy],Ystr[yy,:].T,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)

    # n >= 2
    else:

        # Compute dY/dr
        YPload = Yload.copy()
        YPpot  = Ypot.copy()
        YPshr  = Yshr.copy()
        YPstr  = Ystr.copy()
        for yy in range(0,len(sint)):
            YPload[yy,:] = f_solid.main(sint[yy],Yload[yy,:].T,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)
            YPpot[yy,:]  = f_solid.main(sint[yy],Ypot[yy,:].T,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)
            YPshr[yy,:]  = f_solid.main(sint[yy],Yshr[yy,:].T,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)
            YPstr[yy,:]  = f_solid.main(sint[yy],Ystr[yy,:].T,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)

    if (n != 1):
        # Convert All Y-Solutions to Okubo & Saito Convention
        Yload = Yload / (a_norm * gs_norm)
        Ypot  = Ypot  / (a_norm * gs_norm)
        Yshr  = Yshr  / (a_norm * gs_norm)
        Ystr  = Ystr  / (a_norm * gs_norm)
        YPload = YPload / (a_norm * gs_norm)
        YPpot  = YPpot  / (a_norm * gs_norm)
        YPshr  = YPshr  / (a_norm * gs_norm)
        YPstr  = YPstr  / (a_norm * gs_norm)

    # Define Gamma
    gamma = (4.*piG)/((a_norm**2.)*gs_norm)

    # Compute dL/dK, dL/dmu, and dL/drho (See Table 2 in Okubo & Saito 1983) for I_LL
    Y1 = Yload; YP1 = YPload
    Y2 = Yload; YP2 = YPload
    dI_LL_dK,dI_LL_dmu,dI_LL_drho = okubo_saito_dI.main(n,sint,Y1,Y2,YP1,YP2,piG,a_norm,rho_norm,g_norm)

    # Compute dL/dK, dL/dmu, and dL/drho (See Table 2 in Okubo & Saito 1983) for I_TL
    Y1 = Ypot; YP1 = YPpot
    Y2 = Yload; YP2 = YPload
    dI_TL_dK,dI_TL_dmu,dI_TL_drho = okubo_saito_dI.main(n,sint,Y1,Y2,YP1,YP2,piG,a_norm,rho_norm,g_norm)    

    # Compute dL/dK, dL/dmu, and dL/drho (See Table 2 in Okubo & Saito 1983) for I_LS
    Y1 = Yload; YP1 = YPload
    Y2 = Yshr; YP2 = YPshr
    dI_LS_dK,dI_LS_dmu,dI_LS_drho = okubo_saito_dI.main(n,sint,Y1,Y2,YP1,YP2,piG,a_norm,rho_norm,g_norm)

    # Compute dL/dK, dL/dmu, and dL/drho (See Table 2 in Okubo & Saito 1983) for I_TT
    Y1 = Ypot; YP1 = YPpot
    Y2 = Ypot; YP2 = YPpot
    dI_TT_dK,dI_TT_dmu,dI_TT_drho = okubo_saito_dI.main(n,sint,Y1,Y2,YP1,YP2,piG,a_norm,rho_norm,g_norm)

    # Compute dL/dK, dL/dmu, and dL/drho (See Table 2 in Okubo & Saito 1983) for I_TS
    Y1 = Ypot; YP1 = YPpot
    Y2 = Yshr; YP2 = YPshr
    dI_TS_dK,dI_TS_dmu,dI_TS_drho = okubo_saito_dI.main(n,sint,Y1,Y2,YP1,YP2,piG,a_norm,rho_norm,g_norm)

    # Compute dL/dK, dL/dmu, and dL/drho (See Table 2 in Okubo & Saito 1983) for I_SS
    Y1 = Yshr; YP1 = YPshr
    Y2 = Yshr; YP2 = YPshr
    dI_SS_dK,dI_SS_dmu,dI_SS_drho = okubo_saito_dI.main(n,sint,Y1,Y2,YP1,YP2,piG,a_norm,rho_norm,g_norm)

    # Compute dL/dK, dL/dmu, and dL/drho (See Eq 20 in Okubo & Endo 1986) for I_LSt
    Y1 = Yload; YP1 = YPload
    Y2 = Ystr; YP2 = YPstr
    dI_LSt_dK,dI_LSt_dmu,dI_LSt_drho = okubo_saito_dI.main(n,sint,Y1,Y2,YP1,YP2,piG,a_norm,rho_norm,g_norm)

    # Compute dL/dK, dL/dmu, and dL/drho (See Eq 20 in Okubo & Endo 1986) for I_StSt
    Y1 = Ystr; YP1 = YPstr
    Y2 = Ystr; YP2 = YPstr
    dI_StSt_dK,dI_StSt_dmu,dI_StSt_drho = okubo_saito_dI.main(n,sint,Y1,Y2,YP1,YP2,piG,a_norm,rho_norm,g_norm)

    # Compute Partial Derivatives of Tide Love Numbers
    dht_dK  = dI_TT_dK - dI_TL_dK
    dlt_dK  = dI_TS_dK.copy()
    dkt_dK  = dI_TT_dK.copy()
    dht_dmu = dI_TT_dmu - dI_TL_dmu
    dlt_dmu = dI_TS_dmu.copy()
    dkt_dmu = dI_TT_dmu.copy()
    dht_drho = dI_TT_drho - dI_TL_drho + gamma*hpot*np.power(sint,2.)
    if (n != 0):
        dlt_drho = dI_TS_drho + gamma*(nlpot/n)*np.power(sint,2.)
    dkt_drho = dI_TT_drho.copy()

    # Compute Partial Derivatives of Load Love Numbers
    dhl_dK  = (dI_TL_dK - dI_LL_dK) 
    dll_dK  = dI_LS_dK.copy() 
    dkl_dK  = dI_TL_dK.copy()
    dhl_dmu = dI_TL_dmu - dI_LL_dmu
    dll_dmu = dI_LS_dmu.copy()
    dkl_dmu = dI_TL_dmu.copy()
    dhl_drho = dI_TL_drho - dI_LL_drho + gamma*(2.*hload - hpot)*np.power(sint,2.)
    if (n != 0):
        dll_drho = dI_LS_drho + gamma*((nlload/n) - hshr)*np.power(sint,2.)
    dkl_drho = dI_TL_drho - gamma*hpot*np.power(sint,2.)

    # Compute Partial Derivatives of Shear Love Numbers
    dhs_dK  = dI_TS_dK - dI_LS_dK
    dls_dK  = dI_SS_dK.copy()
    dks_dK  = dI_TS_dK.copy()
    dhs_dmu = dI_TS_dmu - dI_LS_dmu
    dls_dmu = dI_SS_dmu.copy()
    dks_dmu = dI_TS_dmu.copy()
    if (n != 0):
        dhs_drho = dI_TS_drho - dI_LS_drho + gamma*((nlpot/n) - (nlload/n) + hshr)*np.power(sint,2.)
        dls_drho = dI_SS_drho + 2*gamma*(nlshr/n)*np.power(sint,2.)
        dks_drho = dI_TS_drho + gamma*(nlpot/n)*np.power(sint,2.)

    # Special Case of n=1 (Okubo & Endo 1986)
    if (n==1):
       
        # LOAD
        dhl_dK = -1.*dI_LL_dK
        dhl_dmu = -1.*dI_LL_dmu 
        dhl_drho = -1.*dI_LL_drho + 2.*gamma*hload*np.power(sint,2.)
        dll_dK = dI_LSt_dK - dI_LL_dK
        dll_dmu = dI_LSt_dmu - dI_LL_dmu
        dll_drho = dI_LSt_drho - dI_LL_drho + 2.*gamma*(nlload/n)*np.power(sint,2.) 
        # For some reason, dI_TL_drho is large. Neglect anyway, since we are not considering tide solutions here & k_1' is always zero. 
        dkl_drho = dkl_dmu.copy() 
        
        # STRESS
        dhs_dK = -1.*dI_LSt_dK
        dhs_dmu = -1.*dI_LSt_dmu
        dhs_drho = -1.*dI_LSt_drho + 2.*gamma*hstr*np.power(sint,2.)
        dls_dK = dI_StSt_dK - dI_LSt_dK
        dls_dmu = dI_StSt_dmu - dI_LSt_dmu
        dls_drho = dI_StSt_drho - dI_LSt_drho + 2.*gamma*nlstr*np.power(sint,2.)        
        # For some reason, dI_TS_drho is large. Neglect anyway, since we are not considering tide solutions here & k_1'' is always zero.
        dks_drho = dks_dmu.copy()

        # TEST
        dll_dK = dhl_dK - dhs_dK
        dll_dmu = dhl_dmu - dhs_dmu
        dll_drho = dhl_drho - dhs_drho

    # Special Case of n=0 
    if (n==0):

        dhl_dmu = -1.*dI_LL_dmu
        dhl_dK  = -1.*dI_LL_dK
        dhl_drho = -1.*dI_LL_drho + gamma*(2.*hload)*np.power(sint,2.)
        dll_dmu = np.zeros((len(dI_LL_dmu),))
        dll_dK  = np.zeros((len(dI_LL_dK),))
        dll_drho = np.zeros((len(dI_LL_drho),))
        dkl_dmu = dll_dmu.copy()
        dkl_dK = dll_dmu.copy()
        dkl_drho = dll_dmu.copy()
        # Tide and Shear Solutions are Undefined: Set to Zero
        dht_dmu = dht_dK = dht_drho = dlt_dmu = dlt_dK = dlt_drho = dkt_dmu = dkt_dK = dkt_drho = np.zeros((len(dhl_dmu),))
        dhs_dmu = dhs_dK = dhs_drho = dls_dmu = dls_dK = dls_drho = dks_dmu = dks_dK = dks_drho = np.zeros((len(dhl_dmu),))

    # Return Variables
    return dht_dmu,dlt_dmu,dkt_dmu,dht_dK,dlt_dK,dkt_dK,dht_drho,dlt_drho,dkt_drho,\
        dhl_dmu,dll_dmu,dkl_dmu,dhl_dK,dll_dK,dkl_dK,dhl_drho,dll_drho,dkl_drho,\
        dhs_dmu,dls_dmu,dks_dmu,dhs_dK,dls_dK,dks_dK,dhs_drho,dls_drho,dks_drho


