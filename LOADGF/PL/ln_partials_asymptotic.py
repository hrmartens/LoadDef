# *********************************************************************
# CAUTION: IN DEVELOPMENT!!
#
# FUNCTION TO COMPUTE THE ASYMPTOTIC PARTIAL DERIVATIVES OF LOVE NUMBERS
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
from LOADGF.LN import f_solid_n0
from LOADGF.LN import f_solid
from LOADGF.PL import okubo_saito_dI
from LOADGF.PL import asymptotic_Y_solns

def main(n,sint,hload,nlload,nkload,hpot,nlpot,nkpot,hshr,nlshr,nkshr,hstr,nlstr,nkstr,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m):

    # Normalized Radius and Gravity at Surface
    a_norm = sint[-1]
    gs_norm = interpolate.splev(a_norm,tck_gnd,der=0)

    # If sint Includes the Surface (sint=1), then Average the Surface Value with the Next Shallowest Value 
    # :: Leaving sint=1 Causes Divide-by-Zero Problems Later
    #sint_surface = np.where(sint==1); sint_surface = sint_surface[0]
    #if (len(sint_surface)):
    #    sint_surface = sint_surface[0]
    #    sint_nosurface = np.delete(sint,sint_surface)
    #    sint[sint_surface] = (sint[sint_surface]+max(sint_nosurface)) / 2.
    #    print(':: Warning: Modified radial vector for asymptotic solutions to not include the surface, which is undefined. [ln_partials_asymptotic.py] | n = ', n)

    # Normalized Rho and g at 'sint' Radii
    rho_norm = interpolate.splev(sint,tck_rnd,der=0)
    g_norm   = interpolate.splev(sint,tck_gnd,der=0)

    # Compute Asymptotic Y-Solutions
    Ypot,Yload,Yshr = asymptotic_Y_solns.main(n,sint,tck_lnd,tck_mnd,tck_rnd,tck_gnd,piG) 

    # Compute dY/dr
    YPload = Yload.copy(); YPpot = Ypot.copy(); YPshr = Yshr.copy()
    if (n == 0):
        Yloadn0 = np.empty((len(sint),4)); Ypotn0 = np.empty((len(sint),4)); Yshrn0 = np.empty((len(sint),4))
        Yloadn0[:,0] = Yload[:,0]; Yloadn0[:,1] = Yload[:,1]; Yloadn0[:,2] = Yload[:,4]; Yloadn0[:,3] = Yload[:,5]
        Ypotn0[:,0] = Ypot[:,0]; Ypotn0[:,1] = Ypot[:,1]; Ypotn0[:,2] = Ypot[:,4]; Ypotn0[:,3] = Ypot[:,5]
        Yshrn0[:,0] = Yshr[:,0]; Yshrn0[:,1] = Yshr[:,1]; Yshrn0[:,2] = Yshr[:,4]; Yshrn0[:,3] = Yshr[:,5]
        YPloadn0 = Yloadn0.copy(); YPpotn0 = Ypotn0.copy(); YPshrn0 = Yshrn0.copy()
        for yy in range(0,len(sint)):
            YPloadn0[yy,:] = f_solid_n0.main(sint[yy],Yloadn0[yy,:].T,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG)
            YPpotn0[yy,:]  = f_solid_n0.main(sint[yy],Ypotn0[yy,:].T,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG)
            YPshrn0[yy,:]  = f_solid_n0.main(sint[yy],Yshrn0[yy,:].T,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG)
        YPload[:,0] = YPloadn0[:,0]; YPload[:,1] = YPloadn0[:,1]; YPload[:,4] = YPloadn0[:,2]; YPload[:,5] = YPloadn0[:,3]
        YPpot[:,0] = YPpotn0[:,0]; YPpot[:,1] = YPpotn0[:,1]; YPpot[:,4] = YPpotn0[:,2]; YPpot[:,5] = YPpotn0[:,3]
        YPshr[:,0] = YPshrn0[:,0]; YPshr[:,1] = YPshrn0[:,1]; YPshr[:,4] = YPshrn0[:,2]; YPshr[:,5] = YPshrn0[:,3] 
        YPload[:,2] = YPload[:,3] = np.zeros((len(sint),)); YPpot[:,2] = YPpot[:,3] = np.zeros((len(sint),))
        YPshr[:,2] = YPshr[:,3] = np.zeros((len(sint),))
    else:
        for yy in range(0,len(sint)):
            YPload[yy,:] = f_solid.main(sint[yy],Yload[yy,:].T,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)
            YPpot[yy,:]  = f_solid.main(sint[yy],Ypot[yy,:].T,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)
            YPshr[yy,:]  = f_solid.main(sint[yy],Yshr[yy,:].T,n,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,m)

    # Define Gamma
    gamma = (4.*piG)/((a_norm**2.)*gs_norm)

    # Compute dI/dK, dI/dmu, and dI/drho (See Table 2 in Okubo & Saito 1983) for I_LL
    Y1 = Yload; YP1 = YPload
    Y2 = Yload; YP2 = YPload
    dI_LL_dK,dI_LL_dmu,dI_LL_drho = okubo_saito_dI.main(n,sint,Y1,Y2,YP1,YP2,piG,a_norm,rho_norm,g_norm)

    # Compute dI/dK, dI/dmu, and dI/drho (See Table 2 in Okubo & Saito 1983) for I_TL
    Y1 = Ypot; YP1 = YPpot
    Y2 = Yload; YP2 = YPload
    dI_TL_dK,dI_TL_dmu,dI_TL_drho = okubo_saito_dI.main(n,sint,Y1,Y2,YP1,YP2,piG,a_norm,rho_norm,g_norm)    

    # Compute dI/dK, dI/dmu, and dI/drho (See Table 2 in Okubo & Saito 1983) for I_LS
    Y1 = Yload; YP1 = YPload
    Y2 = Yshr; YP2 = YPshr
    dI_LS_dK,dI_LS_dmu,dI_LS_drho = okubo_saito_dI.main(n,sint,Y1,Y2,YP1,YP2,piG,a_norm,rho_norm,g_norm)

    # Compute dI/dK, dI/dmu, and dI/drho (See Table 2 in Okubo & Saito 1983) for I_TT
    Y1 = Ypot; YP1 = YPpot
    Y2 = Ypot; YP2 = YPpot
    dI_TT_dK,dI_TT_dmu,dI_TT_drho = okubo_saito_dI.main(n,sint,Y1,Y2,YP1,YP2,piG,a_norm,rho_norm,g_norm)

    # Compute dI/dK, dI/dmu, and dI/drho (See Table 2 in Okubo & Saito 1983) for I_TS
    Y1 = Ypot; YP1 = YPpot
    Y2 = Yshr; YP2 = YPshr
    dI_TS_dK,dI_TS_dmu,dI_TS_drho = okubo_saito_dI.main(n,sint,Y1,Y2,YP1,YP2,piG,a_norm,rho_norm,g_norm)

    # Compute dI/dK, dI/dmu, and dI/drho (See Table 2 in Okubo & Saito 1983) for I_SS
    Y1 = Yshr; YP1 = YPshr
    Y2 = Yshr; YP2 = YPshr
    dI_SS_dK,dI_SS_dmu,dI_SS_drho = okubo_saito_dI.main(n,sint,Y1,Y2,YP1,YP2,piG,a_norm,rho_norm,g_norm)

    # Compute Partial Derivatives of Asymptotic Tide Love Numbers
    adht_dK  = dI_TT_dK - dI_TL_dK
    adlt_dK  = dI_TS_dK.copy()
    adkt_dK  = dI_TT_dK.copy()
    adht_dmu = dI_TT_dmu - dI_TL_dmu
    adlt_dmu = dI_TS_dmu.copy()
    adkt_dmu = dI_TT_dmu.copy()
    adht_drho = dI_TT_drho - dI_TL_drho + gamma*hpot*np.power(sint,2.)
    if (n == 0):
        adlt_drho = np.zeros((len(adlt_dK),)) # Fill with Zeros
    else:
        adlt_drho = dI_TS_drho + gamma*(nlpot/n)*np.power(sint,2.)
    adkt_drho = dI_TT_drho.copy()

    # Compute Partial Derivatives of Asymptotic Load Love Numbers
    adhl_dK  = (dI_TL_dK - dI_LL_dK) 
    adll_dK  = dI_LS_dK.copy() 
    adkl_dK  = dI_TL_dK.copy()
    adhl_dmu = dI_TL_dmu - dI_LL_dmu
    adll_dmu = dI_LS_dmu.copy()
    adkl_dmu = dI_TL_dmu.copy()
    adhl_drho = dI_TL_drho - dI_LL_drho + gamma*(2.*hload - hpot)*np.power(sint,2.)
    if (n == 0):
        adll_drho = np.zeros((len(adll_dK),)) # Fill with Zeros
    else:
        adll_drho = dI_LS_drho + gamma*((nlload/n) - hshr)*np.power(sint,2.)
    adkl_drho = dI_TL_drho - gamma*hpot*np.power(sint,2.)

    # Compute Partial Derivatives of Asymptotic Shear Love Numbers
    adhs_dK  = dI_TS_dK - dI_LS_dK
    adls_dK  = dI_SS_dK.copy()
    adks_dK  = dI_TS_dK.copy()
    adhs_dmu = dI_TS_dmu - dI_LS_dmu
    adls_dmu = dI_SS_dmu.copy()
    adks_dmu = dI_TS_dmu.copy()
    if (n == 0):
        adhs_drho = np.zeros((len(adls_dK),)) # Fill with Zeros
        adls_drho = adhs_drho.copy()
        adks_drho = adhs_drho.copy()
    else:
        adhs_drho = dI_TS_drho - dI_LS_drho + gamma*((nlpot/n) - (nlload/n) + hshr)*np.power(sint,2.)
        adls_drho = dI_SS_drho + 2*gamma*(nlshr/n)*np.power(sint,2.)
        adks_drho = dI_TS_drho + gamma*(nlpot/n)*np.power(sint,2.)
  
    # Return Variables
    return adht_dmu,adlt_dmu,adkt_dmu,adht_dK,adlt_dK,adkt_dK,adht_drho,adlt_drho,adkt_drho,\
        adhl_dmu,adll_dmu,adkl_dmu,adhl_dK,adll_dK,adkl_dK,adhl_drho,adll_drho,adkl_drho,\
        adhs_dmu,adls_dmu,adks_dmu,adhs_dK,adls_dK,adks_dK,adhs_drho,adls_drho,adks_drho


