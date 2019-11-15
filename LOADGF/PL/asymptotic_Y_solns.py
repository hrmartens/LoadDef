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
from scipy import interpolate

def main(n,sint,tck_lnd,tck_mnd,tck_rnd,tck_gnd,piG):

    # COMPUTE PRELIMINARY PARAMTERS
    # :: Interpolate Non-Dimensional Parameters to 'sint' Radii Values
    rnd = interpolate.splev(sint,tck_rnd)
    lnd = interpolate.splev(sint,tck_lnd)
    mnd = interpolate.splev(sint,tck_mnd)
    gnd = interpolate.splev(sint,tck_gnd)
    # :: Normalized Radius and Gravity at Surface
    a_norm = sint[-1]
    gs_norm = interpolate.splev(a_norm,tck_gnd,der=0)
    # :: Compute Parameters from Okubo (1988)
    gamma = (4.*piG)/((a_norm**2.)*gs_norm)
    alpha = np.sqrt(np.divide((lnd + 2.*mnd),rnd))
    beta = np.sqrt(np.divide(mnd,rnd))
    x = n*(1.-sint) # Normalized Depth (Okubo 1988, between equations 3.2 and 3.3)
    xi = gs_norm/(gamma*a_norm)
    t = (gs_norm*a_norm)/(2.*np.multiply(alpha,beta))
    N = 1. - np.power(np.divide(beta,alpha),2.)
    M = np.divide(np.power(beta,2.),(np.power(alpha,2.) - np.power(beta,2.)))
    D = np.divide((np.power(alpha,2.) + np.power(beta,2.)),(np.power(alpha,2.) - np.power(beta,2.)))
    T = np.divide(1. + np.divide(np.multiply(beta,t),alpha), 1. - np.divide(np.power(beta,2.),np.power(alpha,2.)))
    # :: Perturbation Parameters (Okubo 1988, Eq. 4.1)
    if (n != 0):
        drnd = -np.multiply((n/x),np.divide((rnd - rnd[-1]),rnd[-1])) 
        dmnd = -np.multiply((n/x),np.divide((mnd - mnd[-1]),mnd[-1]))
        dlnd = -np.multiply((n/x),np.divide((lnd - lnd[-1]),lnd[-1]))
        dgnd = gnd - gamma*sint
    else:
        drnd = dmnd = dlnd = dgnd = np.zeros((len(x),))
    dnu = dmnd - drnd/2.
    lndPmnd = lnd + mnd
    if (n != 0):
        dlndPmnd = -np.multiply((n/x),np.divide((lndPmnd - lndPmnd[-1]),lndPmnd[-1]))
    else:
        dlndPmnd = np.zeros((len(x),))
    # :: Compute Additional Parameters from Okubo (1988)
    P = 1. + np.divide(np.multiply(beta,t),alpha) - np.multiply((1. + np.divide(np.power(beta,2.),np.power(alpha,2.))),dnu)
    S = np.multiply(np.divide(np.power(beta,2.),np.power(alpha,2.)),dlndPmnd)
    Q = np.divide(np.multiply(beta,t),alpha) - S + np.multiply((1. + np.divide(np.power(beta,2.),np.power(alpha,2.))),(drnd/2.))

    # TIDE, LOAD, and SHEAR SOLUTIONS (Okubo 1988, Eqs. 4.3-4.5)
    ytide = np.zeros((len(sint),6))
    yload = np.zeros((len(sint),6))
    yshr  = np.zeros((len(sint),6))
    for jj in range(0,len(sint)):
        # :: Parameters at Current Radius
        mys = sint[jj]
        myalpha = alpha[jj]
        mybeta = beta[jj]
        myx = x[jj]
        myxi = xi
        myt = t[jj]
        myN = N[jj]   
        myM = M[jj]
        myD = D[jj]
        myT = T[jj]
        myP = P[jj]
        myS = S[jj]
        myQ = Q[jj]
        mydmnd = dmnd[jj]
        mydrnd = drnd[jj]
        mydgnd = dgnd[jj]
        mydlnd = dlnd[jj]
        mydnu  = dnu[jj]
        myrnd = rnd[jj]
        mymnd = mnd[jj]
        mylnd = lnd[jj]
        mygnd = gnd[jj]
        # :: Compute Asymptotic Y-Solutions (Okubo 1988, Eqs. 4.3-4.5)
        if (n != 0):
            # TIDE (Eq. 4.3)
            ytide[jj,0] = (1./mys)*(1./n)*(a_norm**2.)*(1./(2.*(mybeta**2.)))*(mys**n)* \
                ( 1. + myx + (1./(2.*n))*(-myP*((myx**2.)-(2.*myx)) - (4. + ((2.*myalpha*myt)/mybeta))*(myx+1.) + \
                3.*mydnu*((2.*myx)+1.) + 2. + myT) )
            ytide[jj,1] = (1./(mys**2.))*(myrnd*(a_norm**2.))*(mys**n)* \
                ( myx + (1./(2.*n))*(-myP - (2.*mydnu))*(myx**2.) )
            ytide[jj,2] = (1./mys)*(1./(n**2.))*(a_norm**2.)*(1./(2.*(mybeta**2.)))*(mys**n)* \
                ( myx + (1./(2.*n))*(-(myP*(myx**2.)) + (-((2.*myalpha*myt)/mybeta) + 2.)*myx + \
                mydnu*(2.*myx - 1.) + 4. - myT) )
            ytide[jj,3] = (1./(mys**2.))*(1./n)*myrnd*(a_norm**2.)*(mys**n)* \
                ( myx + (1./(2.*n))*( (-myP - (2.*mydnu))*(myx**2.) - myN*(2.*((myalpha*myt)/mybeta) - mydnu)*myx ) )
            ytide[jj,4] = (mys**n) * ( 1. + (1./(n**2.))*((3.*gamma*(a_norm**2.))/(4.*(mybeta**2.))) + \
                (1./(n**3.))*((3.*gamma*(a_norm**2.))/(4.*(mybeta**2.)))*( ( -(mybeta*myt/myalpha) - (mydrnd/2.) + \
                ((mybeta**2.)/(2.*(myalpha**2.)))*mydnu )*(myx**2.) + \
                (1. - myT - (3./2.)*mydrnd)*myx - 1. - (myalpha*myt/mybeta) + (3./2.)*mydnu ) ) 
            ytide[jj,5] = (1./mys)*(mys**n) * ( 2.*n + 1. - (1./n)*((3.*gamma*(a_norm**2.))/(2.*(mybeta**2.)))*myx )
            # LOAD (Eq. 4.4)
            yload[jj,0] = (1./mys)*((myxi*(a_norm**2.))/(6.*(mybeta**2.)))*(mys**n) * ( -2.*myx - (2./myN) + (1./n)* \
                ( (myP-myQ)*(myx**2.) + 2.*(myalpha*myt/mybeta)*(2.-myD)*(myx+(1./myN)) + (myD - myS - 2.*myN*mydmnd - 2.*mydmnd)*myx + \
                2.*(myM**2.)*(-1. + (myalpha*myt/mybeta)) - (myS/myN) - 3.*mydmnd ) ) + ytide[jj,0]
            yload[jj,1] = (1./(mys**2.))*n*(myxi*myrnd*(a_norm**2.)/3.)*(mys**n) * ( -2.*myx - 2. + (1./n)* \
                ( (myP-myQ + 2.*mydmnd)*(myx**2.) + (5. - 2.*myT)*myx - 1. ) ) + ytide[jj,1]
            yload[jj,2] = (1./mys)*(1./n)*(myxi*(a_norm**2.)/(6.*(mybeta**2.)))*(mys**n) * ( -2.*myx + 2.*myM + (1./n)* \
                ( (myP-myQ)*(myx**2.) + (2.*(myalpha*myt/mybeta) - 3.*myD - 2.*mydmnd)*myx + 1. + ((2.*myT + myS - 6.)/myN) + mydmnd ) ) + ytide[jj,2]
            yload[jj,3] = (1./(mys**2.))*((myxi*myrnd*(a_norm**2.))/3.)*(mys**n)*myx * ( -2. + (1./n)* \
                ( (myP-myQ + 2.*mydmnd)*(myx-2.) + 2.*(myalpha*myt/mybeta)*(1.-myM) + 1. - 2.*myM ) ) + ytide[jj,3]
            yload[jj,4] = (1./n)*(gs_norm*a_norm/(2.*(mybeta**2.)))*(mys**n) * ( 2.*myM*myx - 1. * (1./n)* \
                ( ( -(2.+mydrnd)/myN + myT + myP - myQ + mydmnd + (3./2.)*mydrnd )*(myx**2.) + ( (mybeta*myt*myD/(myalpha*myN)) - 2.*myM + \
                myM*myD + myS/myN + (3./2.)*mydrnd )*myx + (3./2.)*(myalpha*myt/mybeta) + 1. - (1./(2.*myN)) - (myalpha*myt/(2.*mybeta*myN)) - \
                (3./2.)*mydnu ) ) + ytide[jj,4]
            yload[jj,5] = (1./mys)*(gs_norm/(2.*(mybeta**2.)))*(mys**n)*myx * ( 2.*myD + (1./n)* \
                ( (2.*myT - 3. + myS + myN*mydmnd - 4.*myM - myD*mydrnd)*myx - (2.*myalpha*myt/(mybeta*myN))*(myN - 2.*myD + (2./myN)) - myD + \
                ((4.*myM + 2.*myS)/myN) + (4.*mydmnd) ) ) + ytide[jj,5]
            # SHEAR (Eq. 4.5)
            yshr[jj,0] = (1./mys)*(1./n)*(myxi*(a_norm**2.)/(6.*(mybeta**2.)))*(mys**n) * ( -2.*myx - 2.*myM + (1./n)* \
                ( (myP-myQ)*(myx**2.) - (myP-myQ + 2.*mydmnd - 1. + 2.*myT)*(2.*myx - (1./myN)) + (1. + (6./myN))*(myx+1.) - 2. ) )
            yshr[jj,1] = (1./(mys**2.))*(myxi*myrnd*(a_norm**2.)/3.)*(mys**n) * ( -2.*myx + (1./n)* \
                ( (myP-myQ + 2.*mydmnd)*(myx**2.) + (5. - (2.*myalpha*myt/(mybeta*myN)) - 3.*myM)*myx ) )
            yshr[jj,2] = (1./mys)*(1./(n**2.))*(myxi*(a_norm**2.)/(6.*(mybeta**2.)))*(mys**n) * ( -2.*myx + (2./myN) + (1./n)* \
                ( (myP-myQ)*(myx**2.) + (-1. - 6.*myM)*myx - 4.*myD + ((2.*myT + myS)/myN) + mydmnd ) )
            yshr[jj,3] = (1./(mys**2.))*(1./n)*(myxi*myrnd*(a_norm**2.)/3.)*(mys**n) * ( 2. - 2.*myx + (1./n)* \
                ( (myP-myQ + 2.*mydmnd)*(myx**2.) + (3. - 2.*myT - 2.*myS - 2.*myN*mydmnd)*myx - 1. ) )
            yshr[jj,4] = (1./(n**2.))*(gs_norm*a_norm/(2.*(mybeta**2.)))*(mys**n) * ( 2.*myM*myx + (1./n)* \
                ( (myT - myD + myS - ((mybeta**2.)/(myalpha**2.))*mydmnd + (mydrnd/2.) - myM*mydrnd)*(myx**2.) + \
                ((mybeta*myt*myD/(myalpha*myN)) + (myM*(2.-5.*myN) + myS)/myN + (mydrnd/2.))*myx - (myT/2.) + 2. - (mydnu/2.) ) )
            yshr[jj,5] = (1./mys)*(1./n)*(gs_norm/(2.*(mybeta**2.)))*(mys**n)*myx * ( 2.*myD + (1./n)* \
                ( ((2.*mybeta*myt)/(myalpha*myN) - myD - myD*mydrnd + myS + myN*mydmnd)*myx + (4.*myT + 2.*myS - 10.)/myN + 3. + 2.*mydmnd ) )
        else:
            ytide[jj,0] = ytide[jj,1] = ytide[jj,2] = ytide[jj,3] = ytide[jj,4] = ytide[jj,5] = 0
            yload[jj,0] = yload[jj,1] = yload[jj,2] = yload[jj,3] = yload[jj,4] = yload[jj,5] = 0
            yshr[jj,0]  = yshr[jj,1]  = yshr[jj,2]  = yshr[jj,3]  = yshr[jj,4]  = yshr[jj,5]  = 0

    # Test (Okubo 1988, Eq. 5.6(.1), dh_inf/dK)
    eta = lnd + mnd
    dh_inf_dK = np.multiply(((gs_norm**2.)*(a_norm**2.)/(2.*piG*(a_norm**3.)*(eta**2.))),((n+0.5)-1. + 2.*T + S + np.multiply(N,dmnd)))
#    if (n==20):
#        print('dh_inf/dK, Eq. 5.6(.1): ', dh_inf_dK, 'n = ', n)

    # Return Asymptotic Y-Solutions
    return ytide,yload,yshr

    


