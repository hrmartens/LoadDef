#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO PLOT Particle Motion Ellipses (PMEs)
#   USEFUL FOR VISUALIZING OCEAN TIDAL LOADING (OTL)
# 
# Copyright (c) 2014-2022: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
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

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd() + "/../../")
 
# Import Python Modules
import numpy as np
import pygmt
from CONVGF.utility import read_pme_file

#### USER INPUT
 
filename=("../pmes/output/pme_OceanOnly_M2_cm_convgf_GOT410c_PREM.txt")

#### BEGIN CODE

# Initiate Figure
fig = pygmt.Figure()

# Make a mercator map
with pygmt.config(MAP_FRAME_TYPE="fancy+",FONT_ANNOT_PRIMARY="16p",FONT="20p"):
    fig.basemap(region=[-128.0, -110.0, 30.0, 50.0], projection="M15c", frame=["ag", '+t"M@-2@- Ocean Tidal Loading"'])

# Plot the coastlines
fig.coast(region=[-128.0, -110.0, 30.0, 50.0], land="white", water="lightgrey", projection="M15c", shorelines=["1/1p","2/0.1p"], borders=["1/0.5p","2/0.5p"])

# Shift plot origin down by 10cm to plot another map
#fig.shift_origin(yshift="-10c")
 
# Read in the ellipse file
sta,lat,lon,eldir,smmj,smmn,eamp,epha,namp,npha,vamp,vpha = read_pme_file.main(filename)

# Sort Ellipses Based on Semi-Major Axis Length
elidx = np.argsort(smmj)
smmj = smmj[elidx]
smmn = smmn[elidx]
eldir = eldir[elidx]
lat = lat[elidx]
lon = lon[elidx]
vamp = vamp[elidx]

# Adjust size of ellipses
factor = 2.
smmj = np.divide(smmj,factor)
smmn = np.divide(smmn,factor)

# Group information by ellipse
eldata = []
for bb in range(0,len(smmj)):
    eldata.append([lon[bb],lat[bb],vamp[bb],eldir[bb],smmj[bb],smmn[bb]])
 
# Colormap
#pygmt.makecpt(cmap="viridis", series=[min(vamp), max(vamp)])
#pygmt.makecpt(cmap="seis", series=[min(vamp), max(vamp)], reverse=True) 
pygmt.makecpt(cmap="seis", series=[0.0,20.0], reverse=True) 

# Ellipse
# e: ellipse, [[lon, lat, direction, major_axis, minor_axis]]
fig.plot(data=eldata, style="e", cmap=True, pen="1p,black")

# Add colorbar legend
with pygmt.config(FONT_ANNOT_PRIMARY="16p",FONT="18p"):
    fig.colorbar(frame='af+l"Up amplitude (mm)"')
 
# Plot reference ellipse
refsmmj = 8./factor
refsmmn = 2./factor
reflon = 360.0-125.0
refel = [[reflon,33.0,0.0,refsmmj,refsmmn]]
fig.plot(data=refel, style="e", color="white", pen="1p,black")
fig.text(text="8 mm x 2 mm", x=reflon, y=32, font="16p,Helvetica,black") 
 
# Save figure
fig.savefig("./output/OTLmap.pdf")
 
# Show Figure
fig.show()

