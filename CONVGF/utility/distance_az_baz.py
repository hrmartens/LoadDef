# *********************************************************************
# FUNCTION TO COMPUTE ANGULAR DISTANCE and AZIMUTH BETWEEN TWO POINTS
# ON A SPHERE
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
from math import sin, cos, tan, acos, atan, atan2, pi

def distaz2(lat0, lon0, lat1, lon1, b_a):
    t0 = np.multiply((90.-geog2geoc(lat0,b_a)),pi/180.)
    p0 = np.multiply(lon0,pi/180.)
    t1 = np.multiply((90.-geog2geoc(lat1,b_a)),pi/180.) 
    p1 = np.multiply(lon1,pi/180.)
    s0, c0 = np.sin(t0), np.cos(t0)
    s1, c1 = np.sin(t1), np.cos(t1) 
    cD     = np.multiply(np.cos(t0),np.cos(t1)) + \
        np.multiply(np.sin(t0),np.multiply(np.sin(t1),np.cos(p1-p0)))
    Delta  = np.multiply(np.arccos(cD),180./pi)
    Az     = np.multiply(np.arctan2(np.multiply(s0*s1,np.sin(p1-p0)),c1-c0*cD),180./pi)
    Baz    = np.multiply(np.arctan2(np.multiply(s0*s1,np.sin(p0-p1)),c0-c1*cD),180./pi)
    if isinstance(Az,float) == True:  # One Value (Not a List of Load Points)
        if  Az < 0.: Az  += 360.
        if Baz < 0.: Baz += 360.
    else:
        idxAz = np.where(Az < 0.)
        Az[idxAz] = Az[idxAz] + 360.
        idxBaz = np.where(Baz < 0.)
        Baz[idxBaz] = Baz[idxBaz] + 360.
    return Az, Baz, Delta


def geog2geoc(geog,b_a):
    # CAREFUL WITH TAN FUNCTION AT +/- 90-Deg!
    geoc = np.arctan(np.tan(geog*(pi/180.))*(b_a**2.))*(180./pi)
    if isinstance(geoc,float) == False:
        idx90 = np.where(geog == 90.)
        geoc[idx90] = 90.
        idx90 = np.where(geog == -90.)
        geoc[idx90] = -90.
    return geoc

