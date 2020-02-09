# *********************************************************************
# FUNCTION TO GENERATE THE STATION-CENTERED TEMPLATE GRID
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
import scipy as sc
from math import pi

def main(delinc1,delinc2,delinc3,delinc4,delinc5,delinc6,izb,z2b,z3b,z4b,z5b,azinc):

    # -- DEFINITIONS -- 
    # delinc1 = angular distance increment for inner zone (most refined)
    # delinc2 = angular distance increment for zone 2
    # delinc3 = angular distance increment for zone 3
    # delinc4 = angular distance increment for zone 4
    # delinc5 = angular distance increment for zone 5
    # delinc6 = angular distance increment for outer zone (least refined)
    # azinc   = azimuthal increment
    # izb     = inner zone boundary (degrees)
    # z2b     = zone 2 boundary
    # z3b     = zone 3 boundary
    # z4b     = zone 4 boundary
    # z5b     = zone 5 boundary

    # Determine Cell Grid Lines
    inum1 = int((izb/delinc1)+1.)                      # number of increments, inner zone
    ldel1 = np.linspace(0.,izb,num=inum1)              # delta values for inner zone
    inum2 = int((z2b-izb)/delinc2)
    ldel2 = np.linspace(izb+delinc2,z2b,num=inum2)
    inum3 = int((z3b-z2b)/delinc3)
    ldel3 = np.linspace(z2b+delinc3,z3b,num=inum3)
    inum4 = int((z4b-z3b)/delinc4)
    ldel4 = np.linspace(z3b+delinc4,z4b,num=inum4)
    inum5 = int((z5b-z4b)/delinc5)
    ldel5 = np.linspace(z4b+delinc5,z5b,num=inum5)
    inum6 = int((180.-z5b)/delinc6)                     # number of increments, outerzone
    ldel6 = np.linspace(z5b+delinc6,180.,num=inum6)     # delta values for outer zone
    gldel = np.concatenate([ldel1,ldel2,ldel3,ldel4,ldel5,ldel6]) # delta values for inner and outer zones
    inuma = int((360./azinc)+1.)                        # number of azimuthal increments
    lazm  = np.linspace(0.,360.,num=inuma)              # azimuthal values for mesh
    glazm = lazm[0:-1]                                  # don't include last element (same as first element)    

    # Determine Unit Area of Each Cell
    unit_area = []
    all_del = np.concatenate([ldel1,ldel2,ldel3,ldel4,ldel5,ldel6])
    all_del_rad = np.multiply(all_del,(pi/180.))
    azinc_rad = np.multiply(azinc,(pi/180.))
    for ii in range(1,len(all_del_rad)):
        unit_area.append(np.multiply(azinc_rad,\
            np.cos(all_del_rad[ii])-np.cos(all_del_rad[ii-1])))
    unit_area = np.asarray(unit_area)

    # Determine Cell Midpoints
    azm_mdpts = lazm + azinc/2.                        # midpoints between azimuthal gridlines
    azm_mdpts = azm_mdpts[0:-1]                        # don't include value > 360.
    del_mdpts1 = ldel1 + delinc1/2.                    # midpoints between delta gridlines, inner zone
    del_mdpts1 = del_mdpts1[0:-1]                      # don't include value > izb
    del_mdpts2 = ldel2 + delinc2/2.                    # midpoints between delta gridlines, zone-2
    del_mdpts2 = del_mdpts2[0:-1]                      # don't include value > z2b    
    del_mdpts3 = ldel3 + delinc3/2.
    del_mdpts3 = del_mdpts3[0:-1]
    del_mdpts4 = ldel4 + delinc4/2.
    del_mdpts4 = del_mdpts4[0:-1]
    del_mdpts5 = ldel5 + delinc5/2.
    del_mdpts5 = del_mdpts5[0:-1]
    del_mdpts6 = ldel6 + delinc6/2.
    del_mdpts6 = del_mdpts6[0:-1]
    del_mdpt_1to2 = izb + delinc2/2.              # midpoint for cell between inner and zone-2 refinement sections
    del_mdpts1 = np.append(del_mdpts1,del_mdpt_1to2)
    del_mdpt_2to3 = z2b + delinc3/2.
    del_mdpts2 = np.append(del_mdpts2,del_mdpt_2to3)
    del_mdpt_3to4 = z3b + delinc4/2.
    del_mdpts3 = np.append(del_mdpts3,del_mdpt_3to4)
    del_mdpt_4to5 = z4b + delinc5/2.
    del_mdpts4 = np.append(del_mdpts4,del_mdpt_4to5)
    del_mdpt_5to6 = z5b + delinc6/2.
    del_mdpts5 = np.append(del_mdpts5,del_mdpt_5to6)
    ldel = np.concatenate([del_mdpts1,del_mdpts2,del_mdpts3,del_mdpts4,del_mdpts5,del_mdpts6])
    lazm = azm_mdpts 

    # Return Delta/Azm Values for Gridlines and Midpoints
    return gldel,glazm,ldel,lazm,unit_area

