# *********************************************************************
# FUNCTION TO TRANSFORM Z SOLUTIONS TO Y SOLUTIONS (n=0)
# See Smylie (2013), Chapter 3
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

def main(sic,alpha,Z):
    Y = Z
    Y[0] = Z[0]*(sic**(alpha+1)) # Y1
    Y[1] = Z[1]*(sic**alpha)     # Y2
    Y[2] = Z[2]*(sic**(alpha+2)) # Y5
    Y[3] = Z[3]*(sic**(alpha+1)) # Y6
    return Y

