# *********************************************************************
# FUNCTION TO COMPUTE LOAD GREENS FUNCTIONS 
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
import math
import os
import sys
import matplotlib.pyplot as plt
from LOADGF.GF import compute_legendre
from LOADGF.GF import compute_disk_factor
from LOADGF.GF import series_sum

def main(n,a,me,mytheta,h,h_inf,h_inf_p,nl,l_inf,l_inf_p,nk,k_inf,k_inf_p,rf_type,lmda_surface,mu_surface,g_surface,disk_factor,angdist,disk_size,apply_taper=False,max_theta=200.):

    # Note on 'max_theta': Used to set a maximum angular distance, beyond which the LGFs are computed by direct sum 
    #   rather than using the asymptotic Love numbers (Guo, personal communication, 2016)
    # Greater than 180 degrees disables the function (i.e., all angular distances will be computed using asymptotic Love numbers)

    # Convert Theta to Radians
    myt = np.multiply(mytheta,(math.pi/180.))
 
    # Copy Love number arrays (degree-one value will be changed)
    lln_h = h.copy()
    lln_nl = nl.copy()
    lln_nk = nk.copy()
    
    # Compute Sine and Cosine Terms
    mys = np.sin(myt/2.)
    myc = np.cos(myt/2.)
    x   = np.cos(myt)
    dx  = -np.sin(myt)
    
    # Compute the Legendre Polynomials By Recursion Relations
    P,dP,ddP = compute_legendre.main(n,myt)
    if (mytheta == 180.): # Limits at theta=180 evaluated using SymPy
        P = (-1.)**n
        dP = n*0.
        ddP = ((-1.)**(n+1.))*n*(n+1.)/2.

    # Compute Disk Factor for Finite Disk (Not Delta Function)
    alpha = np.radians(disk_size) # Size of disk in radians (value in parentheses given in degrees)
    if disk_factor:
        dfac  = compute_disk_factor.main(n,alpha)

    # Transform the Degree-One Load Love Numbers Based on Requested LGF Type (Blewitt 2003)
    deg_one = np.where(n == 1); deg_one = deg_one[0]
    if (rf_type == 'CE'):
        lln_h[deg_one] = h[deg_one] * 1.
        lln_nl[deg_one] = nl[deg_one] * 1.
        lln_nk[deg_one] = nk[deg_one] * 1.
    elif (rf_type == 'CM'):
        lln_h[deg_one] = h[deg_one] - 1.
        lln_nl[deg_one] = nl[deg_one] - 1.
        lln_nk[deg_one] = nk[deg_one] - 1.
    elif (rf_type == 'CF'):
        lln_h[deg_one] = (2./3.)*(h[deg_one] - nl[deg_one])
        lln_nl[deg_one] = (-1./3.)*(h[deg_one] - nl[deg_one])
        lln_nk[deg_one] = 1. - (1./3.)*h[deg_one] - (2./3.)*nl[deg_one] - 1.

    # Compute coefficients to improve convergence of the LGF series (Guo et al. 2004)
    #  The coefficients effectively "taper" the summands as the series approaches n=n_max
    #  Applying the coefficients mimics the recursive averaging of successive summands 
    #   (reduces amplitude of series oscillations)
    #  The coefficients are similar to binomial coefficients, but in this case we want 
    #   each entry to be the *average* (not sum) of the two above it
    # Note that this computation will be the same for all theta; could pre-compute and pass in
    # This procedure obviates the need for series_sum.py
    recursive_iterations = 200
    cfs = np.ones(len(n))
    conv_coeff = np.ones((recursive_iterations,recursive_iterations+1))
    # Last entry in first row (first recursive iteration) is zero (series truncated here)
    conv_coeff[0,recursive_iterations] = 0.0 
    # Loop through recursive iterations
    for ii in range(1,recursive_iterations):
        # Loop through elements of the series
        for jj in range(0,recursive_iterations):
            # Average consecutive elements
            conv_coeff[ii,jj] = 0.5*(conv_coeff[ii-1,jj]+conv_coeff[ii-1,jj+1])
        # Reset final element in series to zero
        conv_coeff[ii,recursive_iterations] = 0.0 
    # Fill the coefficients into the "convergence factors" array
    # The length of the "convergence factors" array is equal to the number of spherical-harmonic degrees (n) used in the summation (not including asymptotic values)
    # Nearly all the values in the array are equal to 1, except for those values at high n (with a dependence on the number of recursive iterations
    if (apply_taper == True):
        cfs[-recursive_iterations::] = conv_coeff[recursive_iterations-1,0:-1]

    # Precision might be an issue with equations in the form of Guo et al. (2004), Appendix B
    # :: Write equations in a different format 
    # :: B2,B6,B8 from Farrell (1972), Appendix A; B4,B7,B9 from Guo (2016), personal communication; Eq. 16a-16d from Na and Baek (2011)
    if (mytheta == 180.): # Limits for theta=180 computed using SymPy
        eq_B2 = 0.5
        eq_B3 = -0.25
        eq_B4 = math.log(0.5) # equivalent to: eq_B4 = -math.log(2.)
        eq_B5 = 0.
        eq_B6 = 0.
        eq_B7 = 0.
        eq_B8 = 3./8.
        eq_B9 = math.log(2.)/2. + 0.25
    else:
        eq_B2 = 1./(2.*mys)
        eq_B3 = -1./(4.*mys)
        eq_B4 = math.log(1.0/(mys*(1.0+mys)))
        eq_B5 = -myc/(4.*(mys**2))
        eq_B6 = -(myc*(1.+2.*mys))/(2.*mys*(1.+mys))
        eq_B7 = math.log(mys*(1.0-mys)/(myc*myc))/(-dx) - math.log(mys*(myc*myc)/(1.0-mys))*(x/(-dx))
        eq_B8 = (1. + mys + (mys**2))/(4.*(mys**2)*(1.+mys))
        eq_B9 = -math.log(mys*(1.0-mys)/(myc*myc))*x/(dx*dx)+math.log(mys*myc*myc/(1.0-mys))/(dx*dx)\
            -(1.0-2.0*mys)/(2.0*mys)

    # Define Normalization Constant, K
    Kfac = 1E12
 
    # ***** Compute Vertical Displacement Greens Functions ***** #
    # Compute First Two Terms (Analytical Legendre Sums)
    first_term  = h_inf * eq_B2
    second_term = h_inf_p * eq_B4
    # Initialize Array and Counter
    usum = np.zeros(len(n))
    uarg = np.zeros(len(n))
    vcount = 0
    # Begin Loop Through n
    for vv in range(0,len(n)):
        # Current n
        myn = n[vv]
        # Degree-0
        if (myn == 0): 
            if (mytheta <= max_theta):
                usum[vcount] = (lln_h[vv] - h_inf) * P[vv] * cfs[vv]
            else:
                usum[vcount] = lln_h[vv]*P[vv] * cfs[vv]
            if disk_factor:
                if (mytheta >= angdist):
                    usum[vcount] *= dfac[vv]
        # All Other Degrees
        else:
            if (mytheta <= max_theta):
                uarg[vcount] = ((myn*(lln_h[vv]-h_inf) - h_inf_p)/myn) * P[vv]
                usum[vcount] = ((myn*(lln_h[vv]-h_inf) - h_inf_p)/myn) * P[vv] * cfs[vv]
            else:
                usum[vcount] = lln_h[vv]*P[vv] * cfs[vv]
            if disk_factor:
                if (mytheta >= angdist):
                    usum[vcount] *= dfac[vv]
        # Update Counter
        vcount = vcount + 1
    # Compute the Sum of the Series
    #summedu = series_sum.main(usum)
    summedu = np.sum(usum)
    # Compute Greens Functions 
    if (mytheta <= max_theta):
        u = (first_term + second_term + summedu) * (a/me)
    else:
        u = summedu * (a/me)
    u_norm = u * (Kfac*a*myt)
    # Optionally Plot Arguments
    plot_arguments = False
    if (rf_type == 'CE') & (plot_arguments == True):
        if (mytheta == 0.1):
            plt.plot(n,uarg)
            plt.title('vertical displacement series argument | theta = 0.1 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('u_series_terms_0p1deg.pdf',format='pdf')
            plt.clf()
            plt.plot(n[8000:10000],uarg[8000:10000])
            plt.title('vertical displacement series argument | theta = 0.1 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('u_series_terms_0p1deg_highn.pdf',format='pdf')
            plt.clf()
        if (mytheta == 1.):
            plt.plot(n,uarg)
            plt.title('vertical displacement series argument | theta = 1 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('u_series_terms_1deg.pdf',format='pdf')
            plt.clf()
            plt.plot(n[8000:10000],uarg[8000:10000])
            plt.title('vertical displacement series argument | theta = 1 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('u_series_terms_1deg_highn.pdf',format='pdf')
            plt.clf()
        if (mytheta == 10.):
            plt.plot(n,uarg)
            plt.title('vertical displacement series argument | theta = 10 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('u_series_terms_10deg.pdf',format='pdf')
            plt.clf()
            plt.plot(n[8000:10000],uarg[8000:10000])
            plt.title('vertical displacement series argument | theta = 10 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('u_series_terms_10deg_highn.pdf',format='pdf')
            plt.clf()
        if (mytheta == 100.):
            plt.plot(n,uarg)
            plt.title('vertical displacement series argument | theta = 100 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('u_series_terms_100deg.pdf',format='pdf')
            plt.clf()
            plt.plot(n[8000:10000],uarg[8000:10000])
            plt.title('vertical displacement series argument | theta = 100 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('u_series_terms_100deg_highn.pdf',format='pdf')
            plt.clf()
    # ********************************************************** #
	
    # **** Compute Horizontal Displacement Greens Functions **** #
    # Compute First Two Terms (Analytical Legendre Sums)
    first_term  = l_inf * eq_B6
    second_term = l_inf_p * eq_B7
    # Initialize Array and Counter
    vsum = np.zeros(len(n))
    hcount = 0
    # Begin Loop Through n
    for hh in range(0,len(n)):
        # Current n
        myn = n[hh]
        # Degree-0
        if (myn == 0):
            vsum[hcount] = 0. # Start sum at n=1
        # All Other Degrees
        else:
            if (mytheta <= max_theta):
                vsum[hcount] = ((myn*(lln_nl[hh]-l_inf) - l_inf_p)/myn) * (dP[hh]/myn) * cfs[hh]
            else:
                vsum[hcount] = (lln_nl[hh]/myn) * dP[hh] * cfs[hh]
            if disk_factor:
                if (mytheta >= angdist):
                    vsum[hcount] *= dfac[hh]
        # Update Counter
        hcount = hcount + 1
    # Compute the Sum of the Series
    #summedv = series_sum.main(vsum)
    summedv = np.sum(vsum)
    # Compute Greens Functions 
    if (mytheta <= max_theta):
        v = (first_term + second_term + summedv) * (a/me)
    else:
        v = summedv * (a/me)
    v_norm = v * (Kfac*a*myt)
    # ********************************************************** #

    # ** Compute Greens Functions for Indirect Gravity Effect ** #
    # Compute First Two Terms (Analytical Legendre Sums)
    first_term  = k_inf * eq_B2
    second_term = k_inf_p * eq_B4
    third_term  = k_inf * eq_B4
    g_N  = eq_B3 * (g_surface/me)  # Newtonian (direct) effect of gravity (e.g. Farrell 1972)
    # Initialize Array and Counter
    gsum = np.zeros(len(n))
    vcount = 0
    # Begin Loop Through n
    for vv in range(0,len(n)):
        # Current n
        myn = n[vv]
        # Degree-0
        if (myn == 0):
            # Without kummer's: (-g/M)*(k'*P); see Guo et al. 2004; see LoadDef ESS paper supplement Eq. (6)
            # nk' at n=0 = 0. k' at n=0 is undefined. 
            # With kummer's: (-g/M) * (k'-k_inf)*P
            gsum[vcount] = (lln_nk[vv] - k_inf) * P[vv] * cfs[vv]
            if disk_factor:
                if (mytheta >= angdist):
                    gsum[vcount] *= dfac[vv]
        # All Other Degrees
        else:
            if (mytheta <= max_theta):
                gsum[vcount] = (((myn*(lln_nk[vv]-k_inf) - k_inf_p)/myn) * P[vv] * cfs[vv]) \
                    + ((lln_nk[vv]-k_inf)/myn * P[vv] * cfs[vv])
            else:
                gsum[vcount] = (lln_nk[vv]/myn) * (myn + 1.) * P[vv] * cfs[vv]
            if disk_factor:
                if (mytheta >= angdist):
                    gsum[vcount] *= dfac[vv]
        # Update Counter
        vcount = vcount + 1
    # Compute the Sum of the Series
    summedg = np.sum(gsum)
    # Compute Greens Functions 
    if (mytheta <= max_theta):
        g_E = (2.*g_surface*(u/a)) + (first_term + second_term + third_term + summedg) * (-g_surface/me)
    else:
        g_E = (2.*g_surface*(u/a)) + summedg * (-g_surface/me)
    g_E_norm = g_E * (1E18 * (a*myt))
    g_N_norm = g_N * (1E18 * (a*myt))
    # ********************************************************** #
 
    # *** Compute Greens Functions for Indirect Tilt Effect **** #
    # Compute First Two Terms (Analytical Legendre Sums)
    first_term  = h_inf * eq_B5
    second_term = h_inf_p * eq_B6
    third_term  = k_inf * eq_B6
    fourth_term = k_inf_p * eq_B7
    t_N  = eq_B5 * (-1./me) # Newtonian (direct) effect of tilt (e.g. Farrell 1972)
    # Initialize Array and Counter
    tsum = np.zeros(len(n))
    vcount = 0
    # Begin Loop Through n
    for vv in range(0,len(n)):
        # Current n
        myn = n[vv]
        # Degree-0
        if (myn == 0):
            tsum[vcount] = 0 # Sum begins at n=1
        # All Other Degrees
        else:
            if (mytheta <= max_theta):
                tsum[vcount] = (((myn*(lln_h[vv]-h_inf) - h_inf_p)/myn) * dP[vv] * cfs[vv]) \
                    - (((myn*(lln_nk[vv]-k_inf) - k_inf_p)/myn) * (dP[vv]/myn) * cfs[vv])
            else:
                tsum[vcount] = (lln_h[vv] - (lln_nk[vv]/myn)) * dP[vv] * cfs[vv]
            if disk_factor:
                if (mytheta >= angdist):
                    tsum[vcount] *= dfac[vv]
        # Update Counter
        vcount = vcount + 1
    # Compute the Sum of the Series
    summedt = np.sum(tsum)
    # Compute Greens Functions 
    if (mytheta <= max_theta):
        t_E = (1./me) * (first_term + second_term - third_term - fourth_term + summedt)
    else:
        t_E = (1./me) * (summedt)
    t_E_norm = t_E * (Kfac * (a*myt)**2)
    t_N_norm = t_N * (Kfac * (a*myt)**2)
    # ********************************************************** #

    # ************ Compute Strain Greens Functions ************* #
    # See, e.g., Farrell (1972), Agnew (2015), Bos & Scherneck (2013),
    #            Guo et al. (2004), Wang et al. (2012)

    # First Compute e_(theta,theta):
    first_term  = l_inf * eq_B8
    second_term = l_inf_p * eq_B9
    # Initialize Array and Counter
    esum = np.zeros(len(n))
    earg = np.zeros(len(n))
    hcount = 0
    # Begin Loop Through n
    for ee in range(0,len(n)):
        # Current n
        myn = n[ee]
        # Degree-0
        if (myn == 0):
            esum[hcount] = 0.
        # All Other Degrees
        else:
            if (mytheta <= max_theta):
                earg[hcount] = ((myn*(lln_nl[ee]-l_inf) - l_inf_p)/myn) * (ddP[ee]/myn)
                esum[hcount] = ((myn*(lln_nl[ee]-l_inf) - l_inf_p)/myn) * (ddP[ee]/myn) * cfs[ee]
                #esum[hcount] = (lln_nl[ee]-l_inf) * (ddP[ee]/myn)
            else:
                esum[hcount] = lln_nl[ee] * (ddP[ee]/myn) * cfs[ee]
            if disk_factor:
                if (mytheta >= angdist):
                    esum[hcount] *= dfac[ee]
        # Update Counter
        hcount = hcount + 1
    # Compute the Sum of the Series
    #summede = series_sum.main(esum)
    summede = np.sum(esum)
    # Compute Greens Function
    if (mytheta <= max_theta):
        e_tt = (u/a) + (first_term + second_term + summede) * (1./me)
    else:
        e_tt = (u/a) + summede * (1./me)
    #e_tt = (u/a) + (first_term + summede) * (1./me)
 
    # Second Compute e_(lambda,lambda)
    e_ll = (u/a) + (x/(-dx)) * (v/a)
    if (mytheta == 180.):
        # NOTE: The result will be the same as for e_tt as theta-->pi.
        #       The limit as theta-->pi of [cot(theta) * dP(cos(theta))/d(theta)]
        #       is equivalent to the limit as theta-->pi of [d^2P(cos(theta))/d(theta)^2]
        # limit theta->180 [cot(theta) * dP(cos(theta))/d(theta)]
        c_n = ((-1.)**(n+1.))*n*((n+1.)/2.)
        # Compute First Two Terms (Analytical Legendre Sums)
        # Multiplying [cot(theta)] into all three terms of the Kummer's transform equation for horizontal displacement LGF
        first_term  = l_inf * eq_B8 # eq_B8 is the limit as theta->pi of [ SUM_1^n {cot(theta)*(1/n)*dP(cos(theta))/d(theta)} ]
        second_term = l_inf_p * eq_B9 # eq_B9 is the limit as theta-> pi of [ SUM_1^n {cot(theta)*(1/n^2)*dP(cos(theta))/d(theta)} ]
        # Initialize Array and Counter
        ell_vsum = np.zeros(len(n))
        hcount = 0
        # Begin Loop Through n
        for hh in range(0,len(n)):
            # Current n
            myn = n[hh]
            # Degree-0
            if (myn == 0):
                ell_vsum[hcount] = 0. # Start sum at n=1
            # All Other Degrees
            else:
                if (mytheta <= max_theta): # replace dP[hh] with c_n[hh] (to include cotangent [x/-dx])
                    ell_vsum[hcount] = ((myn*(lln_nl[hh]-l_inf) - l_inf_p)/myn) * (c_n[hh]/myn) * cfs[hh]
                else:
                    ell_vsum[hcount] = (lln_nl[hh]/myn) * c_n[hh] * cfs[hh]
                if disk_factor:
                    if (mytheta >= angdist):
                        ell_vsum[hcount] *= dfac[hh]
            # Update Counter
            hcount = hcount + 1
        # Compute the Sum of the Series
        #summedv = series_sum.main(vsum)
        ell_summedv = np.sum(ell_vsum)
        # Compute Greens Functions 
        if (mytheta <= max_theta):
            ell_v = (first_term + second_term + ell_summedv) * (a/me)
        else:
            ell_v = ell_summedv * (a/me)
        ell_v_norm = ell_v * (Kfac*a*myt)
        e_ll = (u/a) + (ell_v/a)
     
    # Third Compute e_(r,r) [e.g., Farrell (1972), Guo et al. (2004)]
    e_rr = (-lmda_surface/(lmda_surface + 2.*mu_surface)) * (e_tt + e_ll)

    # Scale the Strain Load Green's Functions (e.g. SPOTL manual, Guo et al. (2004), Farrell (1972))
    e_tt_norm = e_tt * (Kfac * (a*myt)**2)
    e_ll_norm = e_ll * (Kfac * (a*myt)**2)
    e_rr_norm = e_rr * (1E18 * (a*myt))

    # Optionally Plot Arguments
    plot_arguments = False
    if (rf_type == 'CE') & (plot_arguments == True):
        if (mytheta == 0.1):
            plt.plot(n,earg)
            plt.title('strain (e_tt) series argument | theta = 0.1 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('ett_series_terms_0p1deg.pdf',format='pdf')
            plt.clf()
            plt.plot(n[8000:10000],earg[8000:10000])
            plt.title('strain (e_tt) series argument | theta = 0.1 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('ett_series_terms_0p1deg_highn.pdf',format='pdf')
            plt.clf()
        if (mytheta == 1.):
            plt.plot(n,earg)
            plt.title('strain (e_tt) series argument | theta = 1 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('ett_series_terms_1deg.pdf',format='pdf')
            plt.clf()
            plt.plot(n[8000:10000],earg[8000:10000])
            plt.title('strain (e_tt) series argument | theta = 1 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('ett_series_terms_1deg_highn.pdf',format='pdf')
            plt.clf()
        if (mytheta == 10.):
            plt.plot(n,earg)
            plt.title('strain (e_tt) series argument | theta = 10 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('ett_series_terms_10deg.pdf',format='pdf')
            plt.clf()
            plt.plot(n[8000:10000],earg[8000:10000])
            plt.title('strain (e_tt) series argument | theta = 10 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('ett_series_terms_10deg_highn.pdf',format='pdf')
            plt.clf()
        if (mytheta == 100.):
            plt.plot(n,earg)
            plt.title('strain (e_tt) series argument | theta = 100 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('ett_series_terms_100deg.pdf',format='pdf')
            plt.clf()
            plt.plot(n[8000:10000],earg[8000:10000])
            plt.title('strain (e_tt) series argument | theta = 100 deg')
            plt.ylabel('series argument')
            plt.xlabel('n')
            plt.savefig('ett_series_terms_100deg_highn.pdf',format='pdf')
            plt.clf()

    # ********************************************************** #

    # Return Greens Functions
    return u,v,u_norm,v_norm,g_E,g_E_norm,t_E,t_E_norm,e_tt,e_ll,e_rr,e_tt_norm,e_ll_norm,e_rr_norm,g_N_norm,t_N_norm




# :: NOTES on the Legendre Sums (Alternative Equations)
    # Some of the Legendre Sums may also be found in, e.g.:
    # Farrell (1972, App. 1), Bos & Scherneck (2013, Eq. 1.106), Guo's Code
    # :: Farrell (1972) Legendre Sums
    #eq_A1 = -1./(4.*mys)
    #eq_A2 = 1./(2.*mys)
    #eq_A3 = -myc/(4.*(mys**2))
    #eq_A4 = -(myc*(1.+2.*mys))/(2.*mys*(1.+mys))
    #eq_A5 = (1. + mys + (mys**2))/(4.*(mys**2)*(1.+mys))
    #eq_B2 = eq_A2.copy()
    #eq_B6 = eq_A4.copy()
    #eq_B8 = eq_A5.copy()
    # :: Guo et al. (2004) Legendre Sums
    #x_guo = 1.
    #x = np.cos(myt)
    #dx = -np.sin(myt)
    #l_guo = np.sqrt(1. + x_guo**2 - (2.*x_guo*x))
    #eq_B2 = 1./l_guo
    #eq_B4 = math.log(2. / (l_guo + 1. - x_guo*x))
    #eq_B6 = -(x_guo * (-dx) * ((1./l_guo) + 1.)) / (l_guo + 1. - x_guo*x)
    #eq_B7 = (1./(-dx)) * math.log( (((x_guo - l_guo + 1.)**2) * (1. - x)) / \
    #    (2.*(l_guo - 1. + x_guo*x)*(1. + x)) ) - \
    #    (x/(-dx)) * math.log( ((x_guo**2)*(dx**2)) / (2.*(l_guo - 1. + x_guo*x)) )
    #eq_B8 = ( (-l_guo*x_guo*x*(l_guo + 1. - x_guo*x) + (x_guo**2)*(dx**2)*(1 + l_guo)) \
    #    / (l_guo * (l_guo + 1. - x_guo*x)**2 ) ) * ((1./l_guo) + 1.) + \
    #    ((x_guo**2)*(dx**2))/((l_guo**3)*(l_guo + 1. - x_guo*x)) 
    ## Note: There was a typo in the Guo et al. (2004) paper, Eq. B9: (x_guo*(x-l_guo))-->(x_guo*(x-1))
    ##   in the fourth line (Also, precision is an issue in this format)
    #eq_B9 = (-x/(dx**2)) * math.log( (((x_guo - l_guo + 1.)**2) * (1. - x)) / \
    #    (2.*(l_guo - 1. + x_guo*x)*(1. + x)) ) + \
    #    (1./dx**2) * math.log( ((x_guo**2)*(dx**2)) / (2.*(l_guo - 1. + x_guo*x)) ) + \
    #    ((x_guo*(x - 1.)*(1. - l_guo))/((l_guo - 1. + x_guo*x)*l_guo)) - \
    #    ((2.*x_guo)/((x_guo - l_guo + 1.)*l_guo)) + 2.
    #eq_B9_1 = (-x/(dx**2)) * math.log( (((x_guo - l_guo + 1.)**2) * (1. - x)) / \
    #    (2.*(l_guo - 1. + x_guo*x)*(1. + x)) )
    #eq_B9_2 = (1./dx**2) * math.log( ((x_guo**2)*(dx**2)) / (2.*(l_guo - 1. + x_guo*x)) )
    #eq_B9_3 = ((x_guo*(x - 1.)*(1. - l_guo))/((l_guo - 1. + x_guo*x)*l_guo)) - \
    #    ((2.*x_guo)/((x_guo - l_guo + 1.)*l_guo)) + 2.

