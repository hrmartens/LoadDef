# *********************************************************************
# FUNCTION TO CONVERT IN-PHASE AND QUADRATURE COMPONENTS TO AMPLITUDE AND PHASE
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

def main(ec1,ec2,nc1,nc2,vc1,vc2):

    eamp = np.sqrt( np.square(ec1) + np.square(ec2) )
    namp = np.sqrt( np.square(nc1) + np.square(nc2) )
    vamp = np.sqrt( np.square(vc1) + np.square(vc2) )
    epha = np.arctan2( ec2, ec1 )
    npha = np.arctan2( nc2, nc1 )
    vpha = np.arctan2( vc2, vc1 )

    # Convert Phases to Degrees
    epha = np.degrees(epha)
    npha = np.degrees(npha)
    vpha = np.degrees(vpha)

    # Convert Amplitudes to mm
    eamp = eamp * 1000.
    namp = namp * 1000.
    vamp = vamp * 1000.

    # If Phase < 0, Add 360.
    if (epha < 0.):
        epha = epha + 360.
    if (npha < 0.): 
        npha = npha + 360.
    if (vpha < 0.):
        vpha = vpha + 360.
  
    # Return Amplitude and Phase for All 3 Components
    return eamp,epha,namp,npha,vamp,vpha

