# *********************************************************************
# FUNCTION TO READ IN A LOVE NUMBER FILE
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

import numpy as np

def main(myfile):

    # Read in the Love Numbers
    n,h,nl,nk = np.loadtxt(myfile,usecols=(0,1,2,3),skiprows=14,unpack=True)

    # Read the Meta-Data
    f = open(myfile,'r')
    for ii,line in enumerate(f):
        if (ii == 6): # Read the Material Parameters
            values = line.split()
            a = float(values[0]); me = float(values[1]); lmda_surface = float(values[2]); mu_surface = float(values[3]); g_surface = float(values[4])
        if (ii == 10): # Read the Asymptotic Values
            values = line.split()
            hp = float(values[0]); hpp = float(values[1]); lp = float(values[2])
            lpp = float(values[3]); kp = float(values[4]); kpp = float(values[5])
            break # Exit Loop

    # Return the Data
    return n,h,nl,nk,hp,hpp,lp,lpp,kp,kpp,a,me,lmda_surface,mu_surface,g_surface

