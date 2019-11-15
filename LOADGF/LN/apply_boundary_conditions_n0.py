# *********************************************************************
# FUNCTION TO APPLY BOUNDARY CONDITIONS FOR n=0
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
from LOADGF.LN import load_boundary_condition_n0
from LOADGF.LN import stress_boundary_condition_n0

def main(n,Y1,Y2,gnd_s,piG):

    # Apply Load Boundary Condition at Surface
    Y1sol_load,Y2sol_load,Y5sol_load,Y6sol_load,m_load = load_boundary_condition_n0.main(n,Y1,Y2,gnd_s,piG)
  
    # Apply Stress Boundary Condition at Surface (See Okubo & Endo, 1986)
    Y1sol_str,Y2sol_str,Y5sol_str,Y6sol_str,m_str = stress_boundary_condition_n0.main(n,Y1,Y2,gnd_s,piG)

    # Combine Y solutions
    Y_load = [Y1sol_load,Y2sol_load,Y5sol_load,Y6sol_load,np.nan,np.nan]
    Y_str = [Y1sol_str,Y2sol_str,Y5sol_str,Y6sol_str,np.nan,np.nan]

    # Love Numbers for Potential and Shear will be Zero (for Degree-0)
    Y_pot = [0.,0.,0.,0.,np.nan,np.nan]
    Y_shr = [0.,0.,0.,0.,np.nan,np.nan]

    # Model Vectors for Potential and Shear (Want Data to be Zero)
    m_pot = [0.,0.]
    m_shr = [0.,0.]

    # Return Variables
    return m_load,m_pot,m_str,m_shr


