# *********************************************************************
# FUNCTION TO APPLY BOUNDARY CONDITIONS FOR n>=1 
# SPECIAL CASE OF NO GRAVITY
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
from LOADGF.LN import load_boundary_condition_noGrav
from LOADGF.LN import potential_boundary_condition_noGrav
from LOADGF.LN import stress_boundary_condition_noGrav
from LOADGF.LN import shear_boundary_condition_noGrav

def main(n,Y1,Y2,gnd_s,piG):

    # Apply Load Boundary Condition at Surface
    Y1sol_load,Y2sol_load,Y3sol_load,Y4sol_load,m_load = load_boundary_condition_noGrav.main(n,Y1,Y2,gnd_s,piG)

    # Apply Potential Boundary Condition at Surface
    Y1sol_pot,Y2sol_pot,Y3sol_pot,Y4sol_pot,m_pot = potential_boundary_condition_noGrav.main(n,Y1,Y2,gnd_s)

    # Apply Stress Boundary Condition at Surface (Okubo & Endo, 1986)
    Y1sol_str,Y2sol_str,Y3sol_str,Y4sol_str,m_str = stress_boundary_condition_noGrav.main(n,Y1,Y2,gnd_s,piG)

    # Apply Shear Boundary Condition at Surface (See Okubo & Saito 1983, and Saito 1978)
    Y1sol_shr,Y2sol_shr,Y3sol_shr,Y4sol_shr,m_shr = shear_boundary_condition_noGrav.main(n,Y1,Y2,gnd_s,piG)

    # Combine Y solutions
    Y_load = [Y1sol_load,Y2sol_load,Y3sol_load,Y4sol_load]
    Y_pot = [Y1sol_pot,Y2sol_pot,Y3sol_pot,Y4sol_pot]
    Y_str = [Y1sol_str,Y2sol_str,Y3sol_str,Y4sol_str]
    Y_shr = [Y1sol_shr,Y2sol_shr,Y3sol_shr,Y4sol_shr]

    # Return Model Variables
    return m_load,m_pot,m_str,m_shr

