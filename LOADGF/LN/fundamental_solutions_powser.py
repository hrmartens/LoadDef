# *********************************************************************
# FUNCTION TO COMPUTE STARTING SOLUTIONS USING POWER-SERIES EXPANSIONS
# OF THE EQUATIONS OF MOTION FOR SPHEROIDAL DEFORMATION (n>=1)
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
from scipy import interpolate
 
def main(si,n,tck_lnd,tck_mnd,tck_rnd,wnd,ond,piG,m):
    
    # Interpolate Parameters at Current Radius
    lndi = interpolate.splev(si,tck_lnd,der=0)
    rndi = interpolate.splev(si,tck_rnd,der=0)
    mndi = interpolate.splev(si,tck_mnd,der=0)

    # Compute Additional Non-Dimensional Parameters
    # See Smylie (2013)
    n1 = n*(n+1.)
    bndi = 1./(lndi+2.*mndi)
    dndi = 2.*mndi*(3.*lndi+2.*mndi)*bndi
    endi = 4.*n1*mndi*(lndi+mndi)*bndi - 2.*mndi
    P1nd = 2.*n*(n*(n+2.)*lndi+(n*(n+2.)-1.)*mndi)
    P2nd = n*(n+5.)+n*(n+3.)*(lndi/mndi)
    Q1nd = (n1+n*(n+3.))*lndi + 2.*n*(n+1)*mndi
    Q2nd = 2.*(n+1.)+(n+3.)*(lndi/mndi)
    gamnd = (4./3.)*piG*rndi - (2./3.)*(ond**2)
 
    # Form the Coefficient Matrix
    # Smylie (2013), PG. 257
    c11 = 2.*lndi*bndi + n + 3.
    c12 = -bndi
    c13 = -n1*lndi*bndi
    c14 = 0.
    c15 = 0.
    c16 = 0.
    c21 = -2.*dndi
    c22 = 4.*mndi*bndi+n+2.
    c23 = n1*dndi
    c24 = -n1
    c25 = 0.
    c26 = 0.
    c31 = 1.
    c32 = 0.
    c33 = n+2.
    c34 = -1./mndi
    c35 = 0.
    c36 = 0.
    c41 = dndi
    c42 = lndi*bndi
    c43 = -endi
    c44 = n+5.
    c45 = 0.
    c46 = 0.
    c51 = -3.*gamnd-2.*(ond**2)
    c52 = 0.
    c53 = 0.
    c54 = 0.
    c55 = n+4.
    c56 = -1.
    c61 = 0.
    c62 = 0.
    c63 = n1*(3.*gamnd+2.*(ond**2))
    c64 = 0.
    c65 = -n1
    c66 = n+5.

    A = np.array([[c11, c12, c13, c14, c15, c16], \
	[c21, c22, c23, c24, c25, c26], \
	[c31, c32, c33, c34, c35, c36], \
	[c41, c42, c43, c44, c45, c46], \
	[c51, c52, c53, c54, c55, c56], \
	[c61, c62, c63, c64, c65, c66]])

    Aprime = np.array([[c11+2., c12, c13, c14, c15, c16], \
        [c21, c22+2., c23, c24, c25, c26], \
        [c31, c32, c33+2., c34, c35, c36], \
        [c41, c42, c43, c44+2., c45, c46], \
        [c51, c52, c53, c54, c55+2., c56], \
        [c61, c62, c63, c64, c65, c66+2.]])

    if (n > 1):

	# Initial Values for First Fundamental Solution (free constant A11)
        A11_A11 = 1.
        A20_A11 = 2.*(n-1)*mndi
        A31_A11 = 1./n
        A40_A11 = 2.*mndi*(n-1.)/n
        A52_A11 = 4.*piG*rndi/n
        A61_A11 = 0.
        Y1a = np.array([[A11_A11, A20_A11, A31_A11, A40_A11, A52_A11, A61_A11]])

        # Initial Values for Second Fundamental Solution (free constant A61)
        A11_A61 = 0.
        A20_A61 = 0.
        A31_A61 = 0.
        A40_A61 = 0.
        A52_A61 = 1./n
        A61_A61 = 1.
        Y2a = np.array([[A11_A61, A20_A61, A31_A61, A40_A61, A52_A61, A61_A61]])

        # Initial Values for Third Fundamental Solution (free constant A40)
        A11_A40 = 1./mndi - n*(P2nd/P1nd)
        A20_A40 = Q2nd - Q1nd*(P2nd/P1nd)
        A31_A40 = P2nd/P1nd
        A40_A40 = 1.
        A52_A40 = (2.*piG*rndi/(2.*n+3.))*((n+3.)*A11_A40 - n1*A31_A40)
        A61_A40 = (n+2.)*A52_A40 - 4.*piG*rndi*A11_A40
        Y3a = np.array([[A11_A40, A20_A40, A31_A40, A40_A40, A52_A40, A61_A40]])

        # Coefficients of Second Terms in Expansion of First Fundamental Solution
        A13_A11 = (rndi/P1nd)*((3.-n)*gamnd + (wnd**2) + (2.*(ond**2)) - (2.*m*wnd*ond/n))*(-n)
        A22_A11 = (rndi/P1nd)*((3.-n)*gamnd + (wnd**2) + (2.*(ond**2)) - (2.*m*wnd*ond/n))*(-Q1nd)
        A33_A11 = (rndi/P1nd)*((3.-n)*gamnd + (wnd**2) + (2.*(ond**2)) - (2.*m*wnd*ond/n))
        A42_A11 = 0.
        A54_A11 = (4.*piG*rndi/(2.*(2.*n+3.)))*((n+3.)*A13_A11 - n1*A33_A11)
        A63_A11 = (n+2.)*A54_A11 - 4.*piG*rndi*A13_A11
        Y1b = np.array([[A13_A11, A22_A11, A33_A11, A42_A11, A54_A11, A63_A11]])	

	# Coefficients of Second Terms in Expansion of Second Fundamental Solution
        A13_A61 = -n*(rndi/P1nd)
        A22_A61 = -(Q1nd/P1nd)*rndi
        A33_A61 = rndi/P1nd
        A42_A61 = 0.
        A54_A61 = (4.*piG*rndi/(2.*(2.*n+3.)))*((n+3.)*A13_A61 - n1*A33_A61)
        A63_A61 = (n+2.)*A54_A61 - 4.*piG*rndi*A13_A61
        Y2b = np.array([[A13_A61, A22_A61, A33_A61, A42_A61, A54_A61, A63_A61]])

        # Coefficients of Second Terms in Expansion of Third Fundamental Solution
        i2 = rndi*(-(4.*gamnd+(wnd**2)+(2.*(ond**2)))*A11_A40 + \
            (n1*gamnd+(2.*m*wnd*ond))*A31_A40 - A61_A40)
        i4 = rndi*((gamnd+(2.*m*wnd*ond/n1))*A11_A40 - \
            ((wnd**2)-(2.*m*wnd*ond/n1))*A31_A40 - A52_A40)
        d_2_3 = np.array([[0., i2, 0., i4, 0., 0.]])
        Y3b = np.linalg.solve(A,d_2_3.T)

        # Coefficients of Third Terms in Expansion of First Fundamental Solution
        i2 = rndi*(-(4.*gamnd+(wnd**2)+(2.*(ond**2)))*A13_A11 + \
            (n1*gamnd+(2.*m*wnd*ond))*A33_A11 - A63_A11)
        i4 = rndi*((gamnd+(2.*m*wnd*ond/n1))*A13_A11 - \
            ((wnd**2)-(2.*m*wnd*ond/n1))*A33_A11 - A54_A11)
        d_3_1 = np.array([[0., i2, 0., i4, 0., 0.]])
        Y1c = np.linalg.solve(A,d_3_1.T)

        # Coefficients of Third Terms in Expansion of Second Fundamental Solution
        i2 = rndi*(-(4.*gamnd+(wnd**2)+(2.*(ond**2)))*A13_A61 + \
            (n1*gamnd+(2.*m*wnd*ond))*A33_A61 - A63_A61)
        i4 = rndi*((gamnd+(2.*m*wnd*ond/n1))*A13_A61 - \
            ((wnd**2)-(2.*m*wnd*ond/n1))*A33_A61 - A54_A61)
        d_3_2 = np.array([[0., i2, 0., i4, 0., 0.]])
        Y2c = np.linalg.solve(A,d_3_2.T)

        # Coefficients of Third Terms in Expansion of Third Fundamental Solution
        i2 = rndi*(-(4.*gamnd+(wnd**2)+(2.*(ond**2)))*Y3b[0] + \
            (n1*gamnd+(2.*m*wnd*ond))*Y3b[2] - Y3b[5])
        i4 = rndi*((gamnd+(2.*m*wnd*ond/n1))*Y3b[0] - \
            ((wnd**2)-(2.*m*wnd*ond/n1))*Y3b[2] - Y3b[4])
        d_3_3 = np.array([[0., float(i2), 0., float(i4), 0., 0.]])
        Y3c = np.linalg.solve(Aprime,d_3_3.T)

    elif (n == 1):

        # Initial Values for First Fundamental Solution
        A10_A10 = 1.
        A2m1_A10 = 0.
        A30_A10 = 1.
        A4m1_A10 = 0.
        A51_A10 = 4.*piG*rndi
        A60_A10 = 0.
        Y1a = np.array([[A10_A10, A2m1_A10, A30_A10, A4m1_A10, A51_A10, A60_A10]])

        # Initial Values for Second Fundamental Solution
        A10_A60 = 0.
        A2m1_A60 = 0.
        A30_A60 = 0.
        A4m1_A60 = 0.
        A51_A60 = 1.
        A60_A60 = 1.
        Y2a = np.array([[A10_A60, A2m1_A60, A30_A60, A4m1_A60, A51_A60, A60_A60]])

        # Initial Values for Third Fundamental Solution
        A12_A41 = (1./mndi)-(P2nd/P1nd)
        A21_A41 = Q2nd - ((Q1nd*P2nd)/P1nd)
        A32_A41 = P2nd/P1nd
        A41_A41 = 1.
        A53_A41 = (4.*piG*rndi/10.)*(4.*A12_A41 - 2.*A32_A41)
        A62_A41 = (3.*A53_A41)-(4.*piG*rndi*A12_A41)
        Y3a = np.array([[A12_A41, A21_A41, A32_A41, A41_A41, A53_A41, A62_A41]])

        # Coefficients of Second Terms in Expansion of First Fundamental Solution
        A12_A10 = (rndi/P1nd)*(2.*gamnd+(wnd**2)+(2.*(ond**2))-(2.*m*wnd*ond))*(-1)
        A21_A10 = (rndi/P1nd)*(2.*gamnd+(wnd**2)+(2.*(ond**2))-(2.*m*wnd*ond))*(-Q1nd)
        A32_A10 = (rndi/P1nd)*(2.*gamnd+(wnd**2)+(2.*(ond**2))-(2.*m*wnd*ond))
        A41_A10 = 0.
        A53_A10 = (4.*piG*rndi/10.)*(4.*A12_A10-2.*A32_A10)
        A62_A10 = (3.*A53_A10)-(4.*piG*rndi*A12_A10)
        Y1b = np.array([[A12_A10, A21_A10, A32_A10, A41_A10, A53_A10, A62_A10]])
	
        # Coefficients of Second Terms in Expansion of Second Fundamental Solution
        A12_A60 = -rndi/P1nd
        A21_A60 = -(rndi*Q1nd)/P1nd
        A32_A60 = rndi/P1nd
        A41_A60 = 0.
        A53_A60 = (4.*piG*rndi/10.)*(4.*A12_A60-2.*A32_A60)
        A62_A60 = (3.*A53_A60)-(4.*piG*rndi*A12_A60)
        Y2b = np.array([[A12_A60, A21_A60, A32_A60, A41_A60, A53_A60, A62_A60]])

        # Coefficients of Second Terms in Expansion of Third Fundamental Solution
        i2 = rndi*(-(4.*gamnd+(wnd**2)+(2.*(ond**2)))*A12_A41 + \
            (2.*gamnd+(2.*m*wnd*ond))*A32_A41 - A62_A41)
        i4 = rndi*((gamnd+(m*wnd*ond))*A12_A41 - ((wnd**2)-(m*wnd*ond))*A32_A41 - A53_A41)
        d_2_3 = np.array([[0., i2, 0., i4, 0., 0.]])
        Y3b = np.linalg.solve(A,d_2_3.T)

        # Coefficients of Third Terms in Expansion of First Fundamental Solution
        i2 = rndi*(-(4.*gamnd+(wnd**2)+(2.*(ond**2)))*A12_A10 + \
            (2.*gamnd+(2.*m*wnd*ond))*A32_A10 - A62_A10)
        i4 = rndi*((gamnd+(m*wnd*ond))*A12_A10 - ((wnd**2)-(m*wnd*ond))*A32_A10 - A53_A10)
        d_3_1 = np.array([[0., i2, 0., i4, 0., 0.]])
        Y1c = np.linalg.solve(A,d_3_1.T)

	# Coefficients of Third Terms in Expansion of Second Fundamental Solution
        i2 = rndi*(-(4.*gamnd+(wnd**2)+(2.*(ond**2)))*A12_A60 + \
            (2.*gamnd+(2.*m*wnd*ond))*A32_A60 - A62_A60)
        i4 = rndi*((gamnd+(m*wnd*ond))*A12_A60 - ((wnd**2)-(m*wnd*ond))*A32_A60 - A53_A60)
        d_3_2 = np.array([[0., i2, 0., i4, 0., 0.]])
        Y2c = np.linalg.solve(A,d_3_2.T)

	# Coefficients of Third Terms in Expansion of Third Fundamental Solution
        i2 = rndi*(-(4.*gamnd+(wnd**2)+(2.*(ond**2)))*Y3b[0] + \
            (2.*gamnd+(2.*m*wnd*ond))*Y3b[2] - Y3b[5])
        i4 = rndi*((gamnd+(m*wnd*ond))*Y3b[0] - ((wnd**2)-(m*wnd*ond))*Y3b[2] - Y3b[4])
        d_3_3 = np.array([[0., float(i2), 0., float(i4), 0., 0.]])
        Y3c = np.linalg.solve(Aprime,d_3_3.T)

    # Combine the Terms
    sol1 = Y1a + Y1b*(si**2) + Y1c.T*(si**4)
    sol2 = Y2a + Y2b*(si**2) + Y2c.T*(si**4)
    sol3 = Y3a + Y3b.T*(si**2) + Y3c.T*(si**4)

    # Form Starting Solution Array in Terms of Z-Variables
    Z = np.zeros((3,6))
    Z[0,:] = sol1
    Z[1,:] = sol2
    Z[2,:] = sol3

    # Convert to Y Variables
    Y = Z.copy()
    alpha = n-2.
    Y[0,0] = Z[0,0]*(si**(alpha+1.))
    Y[0,1] = Z[0,1]*(si**(alpha))
    Y[0,2] = Z[0,2]*(si**(alpha+1.))
    Y[0,3] = Z[0,3]*(si**(alpha))
    Y[0,4] = Z[0,4]*(si**(alpha+2.))
    Y[0,5] = Z[0,5]*(si**(alpha+1.))
    Y[1,0] = Z[1,0]*(si**(alpha+1.))
    Y[1,1] = Z[1,1]*(si**(alpha))
    Y[1,2] = Z[1,2]*(si**(alpha+1.))
    Y[1,3] = Z[1,3]*(si**(alpha))
    Y[1,4] = Z[1,4]*(si**(alpha+2.))
    Y[1,5] = Z[1,5]*(si**(alpha+1.))
    alpha = n
    Y[2,0] = Z[2,0]*(si**(alpha+1.))
    Y[2,1] = Z[2,1]*(si**(alpha))
    Y[2,2] = Z[2,2]*(si**(alpha+1.))
    Y[2,3] = Z[2,3]*(si**(alpha))
    Y[2,4] = Z[2,4]*(si**(alpha+2.))
    Y[2,5] = Z[2,5]*(si**(alpha+1.))

    # Return Y-Variable Staring Solutions
    return Y

